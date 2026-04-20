use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::info;

use crate::task_registry::{TaskRegistry, Task, TaskType, Domain};
use crate::task_embedding::{TaskEmbedding, TaskMatcher};
use crate::audit::AuditChain;

/// 联邦学习轮次状态
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoundState {
    pub round_num: i32,
    pub status: RoundStatus,
    pub target_task_id: String,
    pub participants: Vec<String>,
    pub updates_received: HashMap<String, ClientUpdate>,
    pub aggregation_weights: HashMap<String, f32>,
    pub global_loss: f64,
    pub started_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum RoundStatus {
    Pending,
    Running,
    Aggregating,
    Completed,
    Failed,
}

/// 客户端提交的模型更新
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClientUpdate {
    pub client_id: String,
    pub task_id: String,
    pub round_num: i32,
    pub num_samples: i32,
    pub local_loss: f64,
    pub local_accuracy: f32,
    pub gradient_norm: f32,
    pub model_params_hash: String,
    pub compressed_size_bytes: i64,
}

/// 聚合结果
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AggregationResult {
    pub round_num: i32,
    pub num_participants: usize,
    pub global_loss: f64,
    pub global_accuracy: f32,
    pub contribution_scores: HashMap<String, f32>,
    pub task_similarity_weights: HashMap<String, f32>,
    pub aggregation_method: String,
}

/// 联邦学习服务器
///
/// 核心创新：任务感知聚合（Task-Aware Aggregation）
/// - 传统 FedAvg：所有客户端均匀加权
/// - Embodied-FL：通过 HNSW 计算任务相似度，相似任务获得更高权重
pub struct FedServer {
    current_round: Mutex<Option<RoundState>>,
    round_counter: Mutex<i32>,
    task_registry: Arc<TaskRegistry>,
    task_matcher: Mutex<TaskMatcher>,
    audit: Arc<AuditChain>,
    aggregation_history: Mutex<Vec<AggregationResult>>,
}

impl FedServer {
    pub fn new(task_registry: Arc<TaskRegistry>, audit: Arc<AuditChain>) -> Self {
        let embedder = TaskEmbedding::with_defaults();
        Self {
            current_round: Mutex::new(None),
            round_counter: Mutex::new(0),
            task_registry,
            task_matcher: Mutex::new(TaskMatcher::new(embedder)),
            audit,
            aggregation_history: Mutex::new(Vec::new()),
        }
    }

    /// 初始化任务匹配索引
    pub fn init_task_index(&self) -> Result<usize> {
        let tasks = self.task_registry.list_tasks(None, 10000);
        let mut matcher = self.task_matcher.lock().unwrap();
        let count = matcher.add_tasks(&tasks)?;
        info!("Task index initialized with {} tasks", count);
        Ok(count)
    }

    /// 开始新一轮联邦学习
    pub fn start_round(&self, target_task_id: &str, min_participants: usize) -> Result<RoundState> {
        let target_task = self.task_registry.get_task(target_task_id)
            .ok_or_else(|| anyhow::anyhow!("Task {} not found", target_task_id))?;

        let mut counter = self.round_counter.lock().unwrap();
        *counter += 1;
        let round_num = *counter;

        // 用 HNSW 搜索相似任务，确定参与者和权重
        let matcher = self.task_matcher.lock().unwrap();
        let similar = matcher.search(&target_task, min_participants.max(5))?;

        let participants: Vec<String> = similar.iter().map(|(id, _, _)| id.clone()).collect();
        let weights: HashMap<String, f32> = similar.iter()
            .map(|(id, sim, _)| (id.clone(), *sim))
            .collect();

        drop(matcher);

        let now = chrono::Utc::now().to_rfc3339();
        let round = RoundState {
            round_num,
            status: RoundStatus::Running,
            target_task_id: target_task_id.to_string(),
            participants: participants.clone(),
            updates_received: HashMap::new(),
            aggregation_weights: weights.clone(),
            global_loss: 0.0,
            started_at: now,
            completed_at: None,
        };

        *self.current_round.lock().unwrap() = Some(round.clone());

        self.audit.append(
            "round_start",
            "server",
            &format!("round={}, target={}, participants={}", round_num, target_task_id, participants.len()),
        )?;

        info!("Round {} started: target={}, participants={}", round_num, target_task_id, participants.len());
        Ok(round)
    }

    /// 接收客户端更新
    pub fn submit_update(&self, update: ClientUpdate) -> Result<(bool, Option<AggregationResult>)> {
        let mut round = self.current_round.lock().unwrap();
        let round = round.as_mut().ok_or_else(|| anyhow::anyhow!("No active round"))?;

        if round.status != RoundStatus::Running {
            anyhow::bail!("Round {} is not running (status: {:?})", round.round_num, round.status);
        }

        if !round.participants.contains(&update.client_id) {
            anyhow::bail!("Client {} not in participant list", update.client_id);
        }

        round.updates_received.insert(update.client_id.clone(), update.clone());

        // 记录审计
        self.audit.append(
            "gradient_update",
            &update.client_id,
            &format!("round={}, task={}, samples={}, loss={:.4}, acc={:.4}",
                update.round_num, update.task_id, update.num_samples,
                update.local_loss, update.local_accuracy),
        )?;

        // 检查是否所有参与者都已提交
        let all_received = round.updates_received.len() >= round.participants.len();
        let result = if all_received {
            round.status = RoundStatus::Aggregating;
            let agg = self.aggregate(&round)?;
            round.status = RoundStatus::Completed;
            round.completed_at = Some(chrono::Utc::now().to_rfc3339());
            round.global_loss = agg.global_loss;
            Some(agg)
        } else {
            None
        };

        Ok((all_received, result))
    }

    /// 任务感知聚合
    ///
    /// 核心算法：
    /// 1. 获取目标任务的嵌入向量
    /// 2. 计算每个参与者任务与目标任务的相似度
    /// 3. 用相似度作为聚合权重（替代 FedAvg 的均匀权重）
    /// 4. 计算贡献度分数 = 相似度 × 数据量 × 损失改善
    fn aggregate(&self, round: &RoundState) -> Result<AggregationResult> {
        let target_task = self.task_registry.get_task(&round.target_task_id)
            .ok_or_else(|| anyhow::anyhow!("Target task not found"))?;

        let matcher = self.task_matcher.lock().unwrap();
        let participant_tasks: Vec<Task> = round.participants.iter()
            .filter_map(|cid| {
                self.task_registry.get_tasks_by_client(cid).first().cloned()
            })
            .collect();

        // 计算任务相似度权重
        let sim_weights = matcher.compute_aggregation_weights(&target_task, &participant_tasks);
        drop(matcher);

        // 计算加权全局指标
        let mut total_weight = 0.0f64;
        let mut weighted_loss = 0.0f64;
        let mut weighted_acc = 0.0f64;
        let mut contribution_scores = HashMap::new();

        for (i, client_id) in round.participants.iter().enumerate() {
            let update = match round.updates_received.get(client_id) {
                Some(u) => u,
                None => continue,
            };

            let w = sim_weights.get(i).copied().unwrap_or(1.0 / round.participants.len() as f32) as f64;
            total_weight += w;
            weighted_loss += update.local_loss * w;
            weighted_acc += update.local_accuracy as f64 * w;

            // 贡献度 = 任务相似度 × 数据量权重 × 准确率
            let data_weight = (update.num_samples as f32 / 1000.0).ln().max(1.0);
            let contribution = sim_weights.get(i).copied().unwrap_or(0.0) * data_weight * update.local_accuracy;
            contribution_scores.insert(client_id.clone(), contribution);
        }

        if total_weight > 0.0 {
            weighted_loss /= total_weight;
            weighted_acc /= total_weight;
        }

        let task_sim_map: HashMap<String, f32> = round.participants.iter().enumerate()
            .filter_map(|(i, cid)| sim_weights.get(i).map(|w| (cid.clone(), *w)))
            .collect();

        let result = AggregationResult {
            round_num: round.round_num,
            num_participants: round.updates_received.len(),
            global_loss: weighted_loss,
            global_accuracy: weighted_acc as f32,
            contribution_scores,
            task_similarity_weights: task_sim_map,
            aggregation_method: "task_aware_weighted_avg".to_string(),
        };

        // 记录审计
        self.audit.append(
            "aggregation",
            "server",
            &format!("round={}, participants={}, loss={:.4}, acc={:.4}, method=task_aware",
                result.round_num, result.num_participants, result.global_loss, result.global_accuracy),
        )?;

        // 更新客户端贡献度
        for (cid, score) in &result.contribution_scores {
            let _ = self.task_registry.update_contribution(cid, *score as f64);
        }

        self.aggregation_history.lock().unwrap().push(result.clone());

        info!("Round {} aggregated: loss={:.4}, acc={:.4}", result.round_num, result.global_loss, result.global_accuracy);
        Ok(result)
    }

    /// 获取当前轮次状态
    pub fn get_round_status(&self) -> Option<RoundState> {
        self.current_round.lock().unwrap().clone()
    }

    /// 获取聚合历史
    pub fn get_history(&self, limit: usize) -> Vec<AggregationResult> {
        let history = self.aggregation_history.lock().unwrap();
        history.iter().rev().take(limit).cloned().collect()
    }

    /// 获取全局模型（模拟）
    pub fn get_global_model(&self) -> Result<GlobalModel> {
        let round = self.current_round.lock().unwrap();
        let last_agg = self.aggregation_history.lock().unwrap().last().cloned();
        Ok(GlobalModel {
            version: round.as_ref().map(|r| r.round_num).unwrap_or(0),
            round_num: round.as_ref().map(|r| r.round_num).unwrap_or(0),
            params_hash: last_agg.as_ref()
                .map(|a| format!("agg_r{}", a.round_num))
                .unwrap_or_else(|| "init".to_string()),
            accuracy: last_agg.as_ref().map(|a| a.global_accuracy).unwrap_or(0.0),
            loss: last_agg.as_ref().map(|a| a.global_loss).unwrap_or(0.0),
            num_participants_last_round: last_agg.as_ref().map(|a| a.num_participants).unwrap_or(0),
        })
    }
}

/// 全局模型信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GlobalModel {
    pub version: i32,
    pub round_num: i32,
    pub params_hash: String,
    pub accuracy: f32,
    pub loss: f64,
    pub num_participants_last_round: usize,
}
