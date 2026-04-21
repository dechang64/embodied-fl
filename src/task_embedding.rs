use anyhow::Result;
use crate::hnsw_index::HnswIndex;
use crate::task_registry::{Task, TaskType, Domain};

/// 任务嵌入生成器
///
/// 将任务元数据（类型、领域、传感器模态等）编码为固定维度的向量，
/// 用于 HNSW 相似度搜索，实现任务感知的联邦聚合加权。
pub struct TaskEmbedding {
    dimension: usize,
}

impl TaskEmbedding {
    /// 创建嵌入生成器
    /// dimension: 嵌入维度（默认 32）
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    pub fn with_defaults() -> Self {
        Self::new(32)
    }

    /// 从任务元数据生成嵌入向量
    ///
    /// 嵌入结构（32维）：
    /// [0-6]   任务类型 one-hot（7种）
    /// [7-13]  领域 one-hot（7种）
    /// [14-20] 传感器模态 one-hot（7种）
    /// [21-23] 数据规模归一化（log scale）
    /// [24-26] 环境复杂度（简单/中等/复杂）
    /// [27-29] 实时性要求（低/中/高）
    /// [30-31] 预留
    pub fn embed(&self, task: &Task) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dimension];

        // [0-6] 任务类型 one-hot
        let type_idx = match task.task_type {
            TaskType::Grasping => 0,
            TaskType::Navigation => 1,
            TaskType::Inspection => 2,
            TaskType::Assembly => 3,
            TaskType::Manipulation => 4,
            TaskType::Locomotion => 5,
            TaskType::Interaction => 6,
        };
        vec[type_idx] = 1.0;

        // [7-13] 领域 one-hot
        let domain_idx = match task.domain {
            Domain::Electronics => 7,
            Domain::Automotive => 8,
            Domain::Healthcare => 9,
            Domain::Logistics => 10,
            Domain::Home => 11,
            Domain::Agriculture => 12,
            Domain::Construction => 13,
        };
        vec[domain_idx] = 1.0;

        // [14-20] 传感器模态 one-hot
        let sensor = task.metadata.get("sensor").map(|s| s.as_str()).unwrap_or("rgb");
        let sensor_idx = match sensor {
            "rgb" => 14,
            "depth" => 15,
            "point_cloud" => 16,
            "force_torque" => 17,
            "lidar" => 18,
            "multimodal" => 19,
            _ => 20,
        };
        vec[sensor_idx] = 1.0;

        // [21-23] 数据规模归一化（log scale）
        let log_size = (task.dataset_size as f32).log2() / 20.0; // normalize by ~1M
        vec[21] = log_size.min(1.0);
        vec[22] = (log_size * 0.8).min(1.0); // slight variation
        vec[23] = (log_size * 0.6).min(1.0);

        // [24-26] 环境复杂度
        let complexity = task.metadata.get("complexity").map(|s| s.as_str()).unwrap_or("medium");
        match complexity {
            "simple" => { vec[24] = 1.0; }
            "medium" => { vec[25] = 1.0; }
            "complex" => { vec[26] = 1.0; }
            _ => { vec[25] = 1.0; }
        }

        // [27-29] 实时性要求
        let latency = task.metadata.get("latency").map(|s| s.as_str()).unwrap_or("medium");
        match latency {
            "low" => { vec[27] = 1.0; }
            "medium" => { vec[28] = 1.0; }
            "high" => { vec[29] = 1.0; }
            _ => { vec[28] = 1.0; }
        }

        vec
    }

    /// 计算两个任务的相似度（余弦相似度）
    pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// 获取嵌入维度
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// 任务相似度搜索器
pub struct TaskMatcher {
    index: HnswIndex,
    tasks: Vec<Task>,
    embedder: TaskEmbedding,
}

impl TaskMatcher {
    pub fn new(embedder: TaskEmbedding) -> Self {
        Self {
            index: HnswIndex::with_defaults(embedder.dimension()),
            tasks: Vec::new(),
            embedder,
        }
    }

    /// 添加任务到索引
    pub fn add_task(&mut self, task: &Task) -> Result<()> {
        let embedding = self.embedder.embed(task);
        self.index.insert(&task.task_id, &embedding)?;
        self.tasks.push(task.clone());
        Ok(())
    }

    /// 批量添加任务
    pub fn add_tasks(&mut self, tasks: &[Task]) -> Result<usize> {
        for task in tasks {
            self.add_task(task)?;
        }
        Ok(tasks.len())
    }

    /// 搜索相似任务
    pub fn search(&self, query_task: &Task, k: usize) -> Result<Vec<(String, f32, Task)>> {
        let query_embedding = self.embedder.embed(query_task);
        let raw = self.index.search(&query_embedding, k, k * 4)?;
        let results: Vec<(String, f32, Task)> = raw
            .into_iter()
            .filter_map(|(id, distance)| {
                self.tasks.iter().find(|t| t.task_id == id).map(|t| {
                    let sim = TaskEmbedding::similarity(&query_embedding, &self.embedder.embed(t));
                    (id, sim, t.clone())
                })
            })
            .collect();
        Ok(results)
    }

    /// 根据任务相似度计算聚合权重
    ///
    /// 核心创新：相似任务获得更高聚合权重，
    /// 而不是传统 FedAvg 的均匀加权。
    pub fn compute_aggregation_weights(
        &self,
        target_task: &Task,
        participant_tasks: &[Task],
    ) -> Vec<f32> {
        let query_embedding = self.embedder.embed(target_task);
        let mut weights: Vec<f32> = participant_tasks
            .iter()
            .map(|t| {
                let t_embedding = self.embedder.embed(t);
                TaskEmbedding::similarity(&query_embedding, &t_embedding)
            })
            .collect();

        // Softmax 归一化
        let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = weights.iter().map(|w| (w - max_w).exp()).sum();
        weights.iter_mut().for_each(|w| {
            *w = (*w - max_w).exp() / exp_sum;
        });

        weights
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }
}
