//! Federated Server — P2P local aggregation (no central coordinator).
//!
//! Each node performs FedAvg locally on received updates.
//! Model updates are gossiped to peers via PeerManager.
//!
//! Architecture (P2P):
//!   Node A ←→ Node B ←→ Node C
//!   Each node: gRPC server + peer clients + local FedAvg engine
//!   No master/slave — all nodes are equal.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::info;
use base64::Engine;
use sha2::Digest;

use crate::task_registry::TaskRegistry;
use crate::task_embedding::{TaskEmbedding, TaskMatcher};
use crate::audit::AuditChain;

/// Aggregation strategy for multi-task FL.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum AggregationStrategy {
    FedAvg,
    TaskAware,
    MultiTask,
    FedProx { mu: f32 },
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        AggregationStrategy::TaskAware
    }
}

/// FL round state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoundState {
    pub round_num: i32,
    pub round_id: i64,
    pub status: String,
    pub target_task_id: String,
    pub participants: Vec<String>,
    pub updates_received: HashMap<String, ClientUpdate>,
    pub aggregation_weights: HashMap<String, f64>,
    pub strategy: AggregationStrategy,
    pub global_loss: f64,
    pub started_at: String,
    pub completed_at: Option<String>,
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
    pub model_heads: Vec<String>,
    pub params_b64: String,
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
    registered_clients: Mutex<HashMap<String, ClientInfo>>,
    /// P2P: Global model parameters (f32 vector), maintained locally via FedAvg.
    global_params: Mutex<Vec<f32>>,
    /// P2P: All received updates (for local FedAvg).
    pending_updates: Mutex<Vec<ClientUpdate>>,
}

/// Registered client info.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClientInfo {
    pub client_id: String,
    pub client_name: String,
    pub task_type: String,
    pub registered_at: String,
}

/// Result of submitting an update for aggregation.
#[derive(Debug, Clone)]
pub struct SubmitUpdateResult {
    pub accepted: bool,
    pub current_round: u32,
    pub updates_received: i32,
}

/// Result of triggering aggregation.
#[derive(Debug, Clone)]
pub struct AggregateResult {
    pub success: bool,
    pub round: u32,
    pub num_clients: i32,
    pub strategy: String,
}

impl FedServer {
    pub fn new(
        task_registry: Arc<TaskRegistry>,
        audit: Arc<AuditChain>,
        task_matcher_registry: Arc<TaskRegistry>,
    ) -> Self {
        let task_matcher = TaskMatcher::new(TaskEmbedding::new(32));

        Self {
            current_round: Mutex::new(None),
            aggregation_history: Mutex::new(Vec::new()),
            task_matcher: Mutex::new(task_matcher),
            strategy: Mutex::new(AggregationStrategy::default()),
            round_counter: Mutex::new(0),
            registered_clients: Mutex::new(HashMap::new()),
            global_params: Mutex::new(Vec::new()),
            pending_updates: Mutex::new(Vec::new()),
        }
    }

    /// Register a client.
    pub fn register_client(
        &self,
        client_id: &str,
        client_name: &str,
        task_type: &str,
    ) -> Result<()> {
        let mut clients = self.registered_clients.lock().unwrap();
        clients.insert(
            client_id.to_string(),
            ClientInfo {
                client_id: client_id.to_string(),
                client_name: client_name.to_string(),
                task_type: task_type.to_string(),
                registered_at: chrono::Utc::now().to_rfc3339(),
            },
        );
        Ok(())
    }

    /// Set aggregation strategy.
    pub fn set_strategy(&self, strategy: AggregationStrategy) {
        *self.strategy.lock().unwrap() = strategy;
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
            round_id: round_num as i64,
            status: "running".to_string(),
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
            "server",
            &format!("round:{},participants:{},strategy:{:?}", round_num, participants.len(), strategy),
        )?;

        info!(
            "Round {} started with {} participants (strategy: {:?})",
            round_num,
            participants.len(),
            strategy
        );
        Ok(round)
    }

    /// Receive client update.
    pub fn receive_update(&self, update: ClientUpdate, audit: &AuditChain) -> Result<()> {
        let mut round = self.current_round.lock().unwrap();
        let r = round.as_mut().ok_or_else(|| anyhow::anyhow!("No active round"))?;

        if r.status != "running" {
            anyhow::bail!("Round is not running (status: {})", r.status);
        }

        if update.round_num != r.round_num {
            anyhow::bail!("Round mismatch: expected {}, got {}", r.round_num, update.round_num);
        }

        r.updates_received.insert(update.client_id.clone(), update.clone());

        audit.append(
            "update_received",
            &update.client_id,
            &format!("round:{},samples:{},loss:{:.4}", update.round_num, update.num_samples, update.local_loss),
        )?;

        info!(
            "Update received from {} (round {}, {} samples)",
            update.client_id, update.round_num, update.num_samples
        );
        Ok(())
    }

    /// Record a client update (called from gRPC service).
    pub fn record_update(
        &self,
        client_id: &str,
        round_id: i64,
        task_type: &str,
        num_samples: i32,
        local_loss: f64,
        local_accuracy: f32,
        weights: &[u8],
    ) -> Result<()> {
        let update = ClientUpdate {
            client_id: client_id.to_string(),
            task_id: String::new(),
            round_num: round_id as i32,
            num_samples,
            local_loss,
            local_accuracy,
            gradient_norm: 0.0,
            compression_ratio: 1.0,
            task_type: task_type.to_string(),
            model_heads: vec![],
            params_b64: base64::engine::general_purpose::STANDARD.encode(weights),
            params_hash: hex::encode(&sha2::Sha256::digest(weights).to_vec()),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Store in current round if active
        let mut round = self.current_round.lock().unwrap();
        if let Some(r) = round.as_mut() {
            if r.status == "running" {
                r.updates_received.insert(client_id.to_string(), update.clone());
            }
        }
        drop(round);

        Ok(())
    }

    /// Local aggregation (metrics only — for contribution tracking).
    ///
    /// Phase 1 note: Real parameter aggregation is done by FedCtx.
    /// This method only computes weighted metrics for local records.
    pub fn aggregate(&self, audit: &AuditChain) -> Result<AggregationResult> {
        let mut round = self.current_round.lock().unwrap();
        let r = round.as_mut().ok_or_else(|| anyhow::anyhow!("No active round"))?;

        r.status = "aggregating".to_string();

        let updates: Vec<&ClientUpdate> = r.updates_received.values().collect();
        if updates.is_empty() {
            r.status = "failed".to_string();
            anyhow::bail!("No updates received");
        }

        let t0 = std::time::Instant::now();
        let strategy = r.strategy.clone();

        // Compute aggregation weights
        let weights: Vec<f32> = match &strategy {
            AggregationStrategy::FedAvg => {
                let n = updates.len() as f32;
                updates.iter().map(|_| 1.0 / n).collect()
            }
            _ => {
                // TaskAware / MultiTask / FedProx: uniform for now
                let n = updates.len() as f32;
                updates.iter().map(|_| 1.0 / n).collect()
            }
        };

        // Store weights
        for (i, u) in updates.iter().enumerate() {
            r.aggregation_weights
                .insert(u.client_id.clone(), weights[i] as f64);
        }

        // Compute global metrics (weighted average)
        let global_loss: f64 = updates
            .iter()
            .zip(weights.iter())
            .map(|(u, w)| u.local_loss * *w as f64)
            .sum();

        let global_accuracy: f32 = updates
            .iter()
            .zip(weights.iter())
            .map(|(u, w)| u.local_accuracy as f64 * *w as f64)
            .sum::<f64>() as f32;

        // Per-head accuracy
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

        // Communication cost
        let comm_cost: u64 = updates.iter().map(|u| u.params_b64.len() as u64).sum();

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
        r.status = "completed".to_string();
        r.completed_at = Some(chrono::Utc::now().to_rfc3339());

        self.aggregation_history.lock().unwrap().push(result.clone());

        audit.append(
            "aggregation_complete",
            "server",
            &format!(
                "round:{},participants:{},loss:{:.4},acc:{:.4},strategy:{:?},time:{}ms",
                r.round_num,
                r.updates_received.len(),
                global_loss,
                global_accuracy,
                strategy,
                agg_time
            ),
        )?;

        info!(
            "Round {} aggregated: loss={:.4}, acc={:.4}, time={}ms",
            r.round_num, global_loss, global_accuracy, agg_time
        );

        Ok(result)
    }

    /// Get round state.
    pub fn get_round(&self, round_id: i64) -> Result<RoundState> {
        let round = self.current_round.lock().unwrap();
        round
            .as_ref()
            .filter(|r| r.round_id == round_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Round {} not found", round_id))
    }

    /// Get round updates.
    pub fn get_round_updates(&self, round_id: i64) -> Option<Vec<ClientUpdate>> {
        let round = self.current_round.lock().unwrap();
        round
            .as_ref()
            .filter(|r| r.round_id == round_id)
            .map(|r| r.updates_received.values().cloned().collect())
    }

    /// Get aggregation history.
    pub fn get_history(&self, limit: usize) -> Vec<AggregationResult> {
        self.aggregation_history.lock().unwrap().iter().rev().take(limit).cloned().collect()
    }

    /// Get global model info.
    pub fn get_global_model(&self) -> Result<GlobalModel> {
        let round = self.current_round.lock().unwrap();
        let last_agg = self.aggregation_history.lock().unwrap().last().cloned();
        let strategy = self.strategy.lock().unwrap();

        Ok(GlobalModel {
            version: round.as_ref().map(|r| r.round_num).unwrap_or(0),
            round_num: round.as_ref().map(|r| r.round_num).unwrap_or(0),
            params_hash: last_agg
                .as_ref()
                .map(|a| format!("agg_r{}", a.round_num))
                .unwrap_or_else(|| "init".to_string()),
            accuracy: last_agg.as_ref().map(|a| a.global_accuracy).unwrap_or(0.0),
            loss: last_agg.as_ref().map(|a| a.global_loss).unwrap_or(0.0),
            num_participants_last_round: last_agg
                .as_ref()
                .map(|a| a.num_participants)
                .unwrap_or(0),
            supported_heads: last_agg
                .as_ref()
                .map(|a| a.per_head_accuracy.keys().cloned().collect())
                .unwrap_or_default(),
            strategy: strategy.clone(),
        })
    }

    // ════════════════════════════════════════════════════════════════
    // P2P: Local FedAvg aggregation (no central coordinator)
    // ════════════════════════════════════════════════════════════════

    /// P2P: Submit update for local FedAvg aggregation.
    ///
    /// Stores the update and performs weighted average of parameters.
    pub async fn submit_update_local(
        &self,
        client_id: &str,
        round: u32,
        weight_bytes: &[u8],
        num_samples: i64,
        train_loss: f32,
    ) -> Result<SubmitUpdateResult> {
        // Convert bytes to f32 array
        let parameters: Vec<f32> = weight_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(arr)
            })
            .collect();

        // Store as pending update
        let update = ClientUpdate {
            client_id: client_id.to_string(),
            task_id: String::new(),
            round_num: round as i32,
            num_samples: num_samples as i32,
            local_loss: train_loss as f64,
            local_accuracy: 0.0,
            gradient_norm: 0.0,
            compression_ratio: 1.0,
            task_type: String::new(),
            model_heads: vec![],
            params_b64: base64::engine::general_purpose::STANDARD.encode(weight_bytes),
            params_hash: hex::encode(&sha2::Sha256::digest(weight_bytes).to_vec()),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        self.pending_updates.lock().unwrap().push(update);

        let count = self.pending_updates.lock().unwrap().len() as i32;

        Ok(SubmitUpdateResult {
            accepted: true,
            current_round: round,
            updates_received: count,
        })
    }

    /// P2P: Trigger local FedAvg aggregation on all pending updates.
    pub async fn trigger_aggregation_local(&self) -> Result<AggregateResult> {
        let mut pending = self.pending_updates.lock().unwrap();
        if pending.is_empty() {
            anyhow::bail!("No pending updates to aggregate");
        }

        let n = pending.len() as f32;
        let mut agg_params: Vec<f32> = Vec::new();

        // FedAvg: weighted average by num_samples
        let total_samples: i32 = pending.iter().map(|u| u.num_samples).sum();
        let total_samples = total_samples.max(1) as f32;

        // Determine max param length
        let max_len = pending
            .iter()
            .map(|u| {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(&u.params_b64)
                    .unwrap_or_default();
                bytes.len() / 4
            })
            .max()
            .unwrap_or(0);

        agg_params.resize(max_len, 0.0f32);

        for u in pending.iter() {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&u.params_b64)
                .unwrap_or_default();
            let params: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
                .collect();
            let weight = u.num_samples as f32 / total_samples;
            for (i, &p) in params.iter().enumerate() {
                if i < agg_params.len() {
                    agg_params[i] += p * weight;
                }
            }
        }

        // Store aggregated parameters
        *self.global_params.lock().unwrap() = agg_params.clone();

        let round = self.round_counter.lock().unwrap().clone() as u32;

        // Clear pending updates
        pending.clear();

        info!(
            "Local FedAvg completed: round={}, params_len={}, participants={}",
            round,
            agg_params.len(),
            n as i32
        );

        Ok(AggregateResult {
            success: true,
            round,
            num_clients: n as i32,
            strategy: "FedAvg".to_string(),
        })
    }

    /// P2P: Get global model parameters as bytes.
    pub async fn get_global_model_bytes(&self) -> Result<Vec<u8>> {
        let params = self.global_params.lock().unwrap();
        let mut bytes = Vec::with_capacity(params.len() * 4);
        for p in params.iter() {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        Ok(bytes)
    }

    /// P2P: Get global model parameters as f32 vector.
    pub fn get_global_params(&self) -> Vec<f32> {
        self.global_params.lock().unwrap().clone()
    }

    /// P2P: Set global model parameters (from peer's aggregated model).
    pub fn set_global_params(&self, params: Vec<f32>) {
        *self.global_params.lock().unwrap() = params;
    }

    /// List registered clients.
    pub fn list_registered_clients(&self) -> Vec<ClientInfo> {
        self.registered_clients.lock().unwrap().values().cloned().collect()
    }

    /// Get current round number (0 if no round started).
    pub fn current_round_num(&self) -> i32 {
        self.round_counter.lock().unwrap().clone()
    }
}
