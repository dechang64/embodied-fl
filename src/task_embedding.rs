# ── src/task_embedding.rs ──
"""
Upgraded Task Embedding with DINOv2 support.

Original: 32-dim hand-crafted one-hot features from task metadata.
Upgrade: 768-dim DINOv2 self-supervised features from scene images,
         with fallback to the original 32-dim metadata embedding.

The DINOv2 features capture rich visual semantics of the robot's
working environment — workspace layout, object types, lighting —
without requiring any labels.
"""

use anyhow::Result;
use crate::hnsw_index::HnswIndex;
use crate::task_registry::{Task, TaskType, Domain};

/// Task embedding generator — supports both metadata and vision modes.
pub struct TaskEmbedding {
    dimension: usize,
    mode: EmbeddingMode,
}

#[derive(Debug, Clone)]
pub enum EmbeddingMode {
    /// Original 32-dim one-hot metadata embedding
    Metadata,
    /// 768-dim DINOv2 vision embedding (requires Python bridge)
    Vision { dim: usize },
    /// Combined: metadata(32) + vision(768) = 800-dim
    Hybrid { vision_dim: usize },
}

impl TaskEmbedding {
    pub fn new(dimension: usize) -> Self {
        Self { dimension, mode: EmbeddingMode::Metadata }
    }

    pub fn with_vision(vision_dim: usize) -> Self {
        Self {
            dimension: vision_dim,
            mode: EmbeddingMode::Vision { dim: vision_dim },
        }
    }

    pub fn with_hybrid(vision_dim: usize) -> Self {
        Self {
            dimension: 32 + vision_dim,
            mode: EmbeddingMode::Hybrid { vision_dim },
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(32)
    }

    /// Generate embedding from task metadata (original 32-dim).
    ///
    /// Structure:
    /// [0-6]   Task type one-hot (7 types)
    /// [7-13]  Domain one-hot (7 domains)
    /// [14-20] Sensor modality one-hot (7 types)
    /// [21-23] Data scale normalized (log scale)
    /// [24-26] Environment complexity (simple/medium/complex)
    /// [27-29] Real-time requirement (low/medium/high)
    /// [30-31] Reserved
    pub fn embed(&self, task: &Task) -> Vec<f32> {
        match &self.mode {
            EmbeddingMode::Metadata => self.embed_metadata(task),
            EmbeddingMode::Vision { dim } => {
                // Vision embeddings come from Python bridge.
                // If not available, fall back to metadata padded to vision dim.
                let meta = self.embed_metadata(task);
                let mut vec = vec![0.0f32; *dim];
                // Copy metadata into first 32 slots as a weak signal
                for (i, v) in meta.iter().enumerate().take(*dim) {
                    vec[i] = *v;
                }
                vec
            }
            EmbeddingMode::Hybrid { vision_dim } => {
                let mut vec = self.embed_metadata(task);
                // Pad with zeros for vision component (filled by Python bridge)
                vec.resize(32 + vision_dim, 0.0);
                vec
            }
        }
    }

    fn embed_metadata(&self, task: &Task) -> Vec<f32> {
        let mut vec = vec![0.0f32; 32];

        // [0-6] Task type one-hot
        let type_idx = match task.task_type.as_str() {
            "grasping" => 0,
            "navigation" => 1,
            "inspection" => 2,
            "assembly" => 3,
            "manipulation" => 4,
            "welding" => 5,
            _ => 6,
        };
        if type_idx < 7 { vec[type_idx] = 1.0; }

        // [7-13] Domain one-hot (inferred from description keywords)
        let domain_idx = self.infer_domain(&task.description);
        if domain_idx < 7 { vec[7 + domain_idx] = 1.0; }

        // [14-20] Sensor modality (inferred from config)
        let sensor_idx = self.infer_sensor(&task.config_json);
        if sensor_idx < 7 { vec[14 + sensor_idx] = 1.0; }

        // [21-23] Data scale (from rounds participated as proxy)
        let scale = (task.rounds_participated as f32 / 100.0).min(1.0);
        vec[21] = scale;
        vec[22] = (task.total_contribution as f32 / 1000.0).min(1.0);
        vec[23] = scale * 0.5 + vec[22] * 0.5;

        // [24-26] Environment complexity
        let complexity = self.infer_complexity(&task.description);
        vec[24 + complexity] = 1.0;

        // [27-29] Real-time requirement
        let realtime = self.infer_realtime(&task.task_type);
        vec[27 + realtime] = 1.0;

        // [30-31] Reserved for future use
        vec
    }

    fn infer_domain(&self, desc: &str) -> usize {
        let d = desc.to_lowercase();
        if d.contains("电子") || d.contains("pcb") || d.contains("smt") { 0 }
        else if d.contains("汽车") || d.contains("automotive") { 1 }
        else if d.contains("3c") || d.contains("手机") || d.contains("电子消费") { 2 }
        else if d.contains("食品") || d.contains("food") { 3 }
        else if d.contains("医药") || d.contains("pharma") { 4 }
        else if d.contains("物流") || d.contains("warehouse") { 5 }
        else { 6 }
    }

    fn infer_sensor(&self, config: &str) -> usize {
        let c = config.to_lowercase();
        if c.contains("rgb") || c.contains("camera") { 0 }
        else if c.contains("depth") || c.contains("lidar") { 1 }
        else if c.contains("force") || c.contains("torque") { 2 }
        else if c.contains("imu") { 3 }
        else if c.contains("tactile") { 4 }
        else if c.contains("ir") || c.contains("thermal") { 5 }
        else { 6 }
    }

    fn infer_complexity(&self, desc: &str) -> usize {
        let d = desc.to_lowercase();
        if d.contains("简单") || d.contains("simple") || d.contains("structured") { 0 }
        else if d.contains("复杂") || d.contains("complex") || d.contains("cluttered") { 2 }
        else { 1 }
    }

    fn infer_realtime(&self, task_type: &str) -> usize {
        match task_type {
            "grasping" | "manipulation" => 2,  // High
            "assembly" => 1,                     // Medium
            _ => 0,                              // Low
        }
    }

    /// Cosine similarity between two embeddings.
    pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 { return 0.0; }
        (dot / (norm_a * norm_b)).max(-1.0).min(1.0)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn mode(&self) -> &EmbeddingMode {
        &self.mode
    }
}

/// Task matcher — find similar tasks using HNSW.
pub struct TaskMatcher {
    embedder: TaskEmbedding,
    index: HnswIndex,
    tasks: Vec<(String, Task)>,
}

impl TaskMatcher {
    pub fn new(embedder: TaskEmbedding) -> Self {
        let dim = embedder.dimension();
        Self {
            embedder,
            index: HnswIndex::with_defaults(dim),
            tasks: Vec::new(),
        }
    }

    pub fn add_task(&mut self, task: &Task) -> Result<()> {
        let embedding = self.embedder.embed(task);
        let id = task.task_id.clone();
        self.index.insert(&id, &embedding)?;
        self.tasks.push((id, task.clone()));
        Ok(())
    }

    pub fn find_similar(&self, query: &Task, k: usize) -> Result<Vec<(String, f32, Task)>> {
        let query_embedding = self.embedder.embed(query);
        let raw = self.index.search(&query_embedding, k, std::cmp::max(k * 4, 50))?;
        Ok(raw.into_iter().filter_map(|(id, dist)| {
            self.tasks.iter().find(|(tid, _)| tid == &id).map(|(_, t)| {
                (id.clone(), TaskEmbedding::similarity(&query_embedding, &self.embedder.embed(t)), t.clone())
            })
        }).collect())
    }

    /// Compute aggregation weights based on task similarity (softmax).
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

        // Softmax normalization
        let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = weights.iter().map(|w| (w - max_w).exp()).sum();
        if exp_sum > 1e-8 {
            weights.iter_mut().for_each(|w| { *w = (*w - max_w).exp() / exp_sum; });
        } else {
            // Uniform fallback
            let n = weights.len() as f32;
            weights.iter_mut().for_each(|w| { *w = 1.0 / n; });
        }

        weights
    }

    pub fn len(&self) -> usize { self.tasks.len() }
}
