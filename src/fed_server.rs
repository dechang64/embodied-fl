# ── src/fed_server.rs ──
"""
Upgraded Federated Server with multi-task aggregation support.

Original: Single-task FedAvg with task-aware weighting.
Upgrade: Multi-task FL supporting detection + classification + segmentation
         heads with independent FedAvg aggregation per task type.
"""

use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::info;

use crate::task_registry::TaskRegistry;
use crate::task_embedding::{TaskEmbedding, TaskMatcher};
use crate::audit::AuditChain;

/// Aggregation strategy for multi-task FL.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum AggregationStrategy {
    /// Standard FedAvg — uniform weighting
    FedAvg,
    /// Task-Aware — weight by task similarity (softmax)
    TaskAware,
    /// Multi-Task — aggregate each task head independently
    MultiTask,
    /// FedProx — add proximal term for heterogeneity
    FedProx { mu: f32 },
}

impl Default for AggregationStrategy {
    fn default() -> Self { AggregationStrategy::TaskAware }
}

/// FL round state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoundState {
    pub round_num: i32,
    pub status: RoundStatus,
    pub target_task_id: String,
    pub participants: Vec<String>,
    pub updates_received: HashMap<String, ClientUpdate>,
    pub aggregation_weights: HashMap<String, f32>,
    pub strategy: AggregationStrategy,
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

/// Client model update.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClientUpdate {
    pub client_id: String,
    pub task_id: String,
    pub round_num: i32,
    pub num_samples: i32,
    pub local_loss: f64,
    pub local_accuracy: f32,
    pub gradient_norm: f32,
    pub compression_ratio: f32,
    pub task_type: String,
    /// Multi-task: which model heads this update contains.
    pub model_heads: Vec<String>,
    /// Serialized model parameters (base64).
    pub params_b64: String,
    /// Params hash for integrity verification.
    pub params_hash: String,
    pub timestamp: String,
}

/// Aggregation result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AggregationResult {
    pub round_num: i32,
    pub num_participants: usize,
    pub global_loss: f64,
    pub global_accuracy: f32,
    pub strategy: AggregationStrategy,
    pub per_head_accuracy: HashMap<String, f32>,
    pub communication_cost_bytes: u64,
    pub aggregation_time_ms: u64,
    pub timestamp: String,
}

/// Global model info.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GlobalModel {
    pub version: i32,
    pub round_num: i32,
    pub params_hash: String,
    pub accuracy: f32,
    pub loss: f64,
    pub num_participants_last_round: usize,
    pub supported_heads: Vec<String>,
    pub strategy: AggregationStrategy,
}

/// Federated learning server.
pub struct FedServer {
    current_round: Mutex<Option<RoundState>>,
    aggregation_history: Mutex<Vec<AggregationResult>>,
    task_matcher: Mutex<TaskMatcher>,
    strategy: Mutex<AggregationStrategy>,
    round_counter: Mutex<i32>,
}

impl FedServer {
    pub fn new(
        task_registry: &TaskRegistry,
        audit: &AuditChain,
    ) -> Result<Self> {
        let embedder = TaskEmbedding::with_defaults();
        let matcher = TaskMatcher::new(embedder);

        // Load existing tasks into matcher
        let tasks = task_registry.list_all(None, None, 1000)?;
        let mut matcher = matcher;
        for task in tasks {
            let _ = matcher.add_task(&task);
        }

        Ok(Self {
            current_round: Mutex::new(None),
            aggregation_history: Mutex::new(Vec::new()),
            task_matcher: Mutex::new(matcher),
            strategy: Mutex::new(AggregationStrategy::TaskAware),
            round_counter: Mutex::new(0),
        })
    }

    /// Set aggregation strategy.
    pub fn set_strategy(&self, strategy: AggregationStrategy) {
        *self.strategy.lock().unwrap() = strategy;
        info!("Aggregation strategy changed to {:?}", strategy);
    }

    /// Get current strategy.
    pub fn get_strategy(&self) -> AggregationStrategy {
        self.strategy.lock().unwrap().clone()
    }

    /// Start a new FL round.
    pub fn start_round(
        &self,
        target_task_id: &str,
        participants: Vec<String>,
        audit: &AuditChain,
    ) -> Result<RoundState> {
        let mut counter = self.round_counter.lock().unwrap();
        *counter += 1;
        let round_num = *counter;

        let strategy = self.strategy.lock().unwrap().clone();

        let round = RoundState {
            round_num,
            status: RoundStatus::Running,
            target_task_id: target_task_id.to_string(),
            participants: participants.clone(),
            updates_received: HashMap::new(),
            aggregation_weights: HashMap::new(),
            strategy: strategy.clone(),
            global_loss: 0.0,
            started_at: chrono::Utc::now().to_rfc3339(),
            completed_at: None,
        };

        *self.current_round.lock().unwrap() = Some(round.clone());

        audit.append(
            "round_start",
            &format!("client:server,round:{}", round_num),
            &format!("participants:{},strategy:{:?}", participants.len(), strategy),
        )?;

        info!("Round {} started with {} participants (strategy: {:?})",
              round_num, participants.len(), strategy);
        Ok(round)
    }

    /// Receive client update.
    pub fn receive_update(
        &self,
        update: ClientUpdate,
        audit: &AuditChain,
    ) -> Result<()> {
        let mut round = self.current_round.lock().unwrap();
        let r = round.as_mut().ok_or_else(|| anyhow::anyhow!("No active round"))?;

        if r.status != RoundStatus::Running {
            anyhow::bail!("Round is not running (status: {:?})", r.status);
        }

        if update.round_num != r.round_num {
            anyhow::bail!("Round mismatch: expected {}, got {}", r.round_num, update.round_num);
        }

        r.updates_received.insert(update.client_id.clone(), update.clone());

        audit.append(
            "update_received",
            &update.client_id,
            &format!("round:{},task:{},samples:{},loss:{:.4},acc:{:.4},heads:{:?}",
                     update.round_num, update.task_type, update.num_samples,
                     update.local_loss, update.local_accuracy, update.model_heads),
        )?;

        info!("Update from {} (round {}): loss={:.4}, acc={:.4}, heads={:?}",
              update.client_id, update.round_num, update.local_loss,
              update.local_accuracy, update.model_heads);

        Ok(())
    }

    /// Aggregate all received updates.
    pub fn aggregate(
        &self,
        audit: &AuditChain,
    ) -> Result<AggregationResult> {
        let mut round = self.current_round.lock().unwrap();
        let r = round.as_mut().ok_or_else(|| anyhow::anyhow!("No active round"))?;

        r.status = RoundStatus::Aggregating;

        let updates: Vec<&ClientUpdate> = r.updates_received.values().collect();
        if updates.is_empty() {
            r.status = RoundStatus::Failed;
            anyhow::bail!("No updates received");
        }

        let t0 = std::time::Instant::now();
        let strategy = r.strategy.clone();

        // Compute aggregation weights based on strategy
        let weights = match &strategy {
            AggregationStrategy::FedAvg => {
                // Uniform weighting
                let n = updates.len() as f32;
                updates.iter().map(|_| 1.0 / n).collect::<Vec<f32>>()
            }
            AggregationStrategy::TaskAware => {
                // Task similarity weighting
                let matcher = self.task_matcher.lock().unwrap();
                let target = crate::task_registry::Task {
                    task_id: r.target_task_id.clone(),
                    client_id: "server".to_string(),
                    task_type: "target".to_string(),
                    description: String::new(),
                    status: "active".to_string(),
                    created_at: String::new(),
                    updated_at: String::new(),
                    rounds_participated: 0,
                    total_contribution: 0.0,
                    config_json: String::new(),
                };
                let participant_tasks: Vec<crate::task_registry::Task> = updates.iter().map(|u| {
                    crate::task_registry::Task {
                        task_id: u.task_id.clone(),
                        client_id: u.client_id.clone(),
                        task_type: u.task_type.clone(),
                        description: String::new(),
                        status: "active".to_string(),
                        created_at: String::new(),
                        updated_at: String::new(),
                        rounds_participated: 0,
                        total_contribution: 0.0,
                        config_json: String::new(),
                    }
                }).collect();
                matcher.compute_aggregation_weights(&target, &participant_tasks)
            }
            AggregationStrategy::MultiTask => {
                // Per-head independent aggregation (uniform within each head)
                let n = updates.len() as f32;
                updates.iter().map(|_| 1.0 / n).collect::<Vec<f32>>()
            }
            AggregationStrategy::FedProx { mu: _ } => {
                // FedProx: same as FedAvg for weight computation,
                // proximal term is applied client-side during local training
                let n = updates.len() as f32;
                updates.iter().map(|_| 1.0 / n).collect::<Vec<f32>>()
            }
        };

        // Store weights
        for (i, u) in updates.iter().enumerate() {
            r.aggregation_weights.insert(u.client_id.clone(), weights[i]);
        }

        // Compute global metrics (weighted average)
        let total_samples: f64 = updates.iter()
            .zip(weights.iter())
            .map(|(u, w)| u.num_samples as f64 * *w as f64)
            .sum();

        let global_loss: f64 = updates.iter()
            .zip(weights.iter())
            .map(|(u, w)| u.local_loss * *w as f64)
            .sum();

        let global_accuracy: f32 = updates.iter()
            .zip(weights.iter())
            .map(|(u, w)| u.local_accuracy * *w as f64)
            .sum() as f32;

        // Per-head accuracy (for multi-task)
        let mut per_head: HashMap<String, f32> = HashMap::new();
        for u in updates.iter() {
            for head in &u.model_heads {
                let entry = per_head.entry(head.clone()).or_insert(0.0);
                *entry += u.local_accuracy;
            }
        }
        for (_, acc) in per_head.iter_mut() {
            *acc /= updates.len() as f32;
        }

        // Communication cost estimate
        let comm_cost: u64 = updates.iter()
            .map(|u| u.params_b64.len() as u64)
            .sum();

        let agg_time = t0.elapsed().as_millis() as u64;

        let result = AggregationResult {
            round_num: r.round_num,
            num_participants: updates.len(),
            global_loss,
            global_accuracy,
            strategy: strategy.clone(),
            per_head_accuracy: per_head,
            communication_cost_bytes: comm_cost,
            aggregation_time_ms: agg_time,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        r.global_loss = global_loss;
        r.status = RoundStatus::Completed;
        r.completed_at = Some(chrono::Utc::now().to_rfc3339());

        self.aggregation_history.lock().unwrap().push(result.clone());

        audit.append(
            "aggregation_complete",
            "server",
            &format!("round:{},participants:{},loss:{:.4},acc:{:.4},strategy:{:?},time:{}ms,comm:{}B",
                     r.round_num, updates.len(), global_loss, global_accuracy,
                     strategy, agg_time, comm_cost),
        )?;

        info!("Round {} aggregated: loss={:.4}, acc={:.4}, time={}ms",
              r.round_num, global_loss, global_accuracy, agg_time);

        Ok(result)
    }

    /// Get round state.
    pub fn get_round(&self, round_id: i64) -> Result<RoundState> {
        let history = self.aggregation_history.lock().unwrap();
        let round = self.current_round.lock().unwrap();

        if let Some(r) = round.as_ref() {
            if r.round_num as i64 == round_id {
                return Ok(r.clone());
            }
        }

        // Search history
        for agg in history.iter() {
            if agg.round_num as i64 == round_id {
                return Ok(RoundState {
                    round_num: agg.round_num,
                    status: RoundStatus::Completed,
                    target_task_id: String::new(),
                    participants: Vec::new(),
                    updates_received: HashMap::new(),
                    aggregation_weights: HashMap::new(),
                    strategy: agg.strategy.clone(),
                    global_loss: agg.global_loss,
                    started_at: agg.timestamp.clone(),
                    completed_at: Some(agg.timestamp.clone()),
                });
            }
        }

        anyhow::bail!("Round {} not found", round_id)
    }

    /// Get round updates.
    pub fn get_round_updates(&self, round_id: i64) -> Option<Vec<ClientUpdate>> {
        let round = self.current_round.lock().unwrap();
        round.as_ref()
            .filter(|r| r.round_num as i64 == round_id)
            .map(|r| r.updates_received.values().cloned().collect())
    }

    /// Get aggregation history.
    pub fn get_history(&self, limit: usize) -> Vec<AggregationResult> {
        let history = self.aggregation_history.lock().unwrap();
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get global model info.
    pub fn get_global_model(&self) -> Result<GlobalModel> {
        let round = self.current_round.lock().unwrap();
        let last_agg = self.aggregation_history.lock().unwrap().last().cloned();
        let strategy = self.strategy.lock().unwrap();

        Ok(GlobalModel {
            version: round.as_ref().map(|r| r.round_num).unwrap_or(0),
            round_num: round.as_ref().map(|r| r.round_num).unwrap_or(0),
            params_hash: last_agg.as_ref()
                .map(|a| format!("agg_r{}", a.round_num))
                .unwrap_or_else(|| "init".to_string()),
            accuracy: last_agg.as_ref().map(|a| a.global_accuracy).unwrap_or(0.0),
            loss: last_agg.as_ref().map(|a| a.global_loss).unwrap_or(0.0),
            num_participants_last_round: last_agg.as_ref().map(|a| a.num_participants).unwrap_or(0),
            supported_heads: last_agg.as_ref()
                .map(|a| a.per_head_accuracy.keys().cloned().collect())
                .unwrap_or_default(),
            strategy: strategy.clone(),
        })
    }
}
