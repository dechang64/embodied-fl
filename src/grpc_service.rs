use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::task_registry::TaskRegistry;
use crate::fed_server::FedServer;
use crate::contribution_tracker::ContributionTracker;
use crate::vector_db::VectorDb;
use crate::audit::AuditChain;

pub mod proto {
    tonic::include_proto!("embodiedfl");
}

use proto::{
    federated_service_server::{FederatedService, FederatedServiceServer},
    task_registry_server::{TaskRegistry as TaskRegistryTrait, TaskRegistryServer},
    contribution_service_server::{ContributionService, ContributionServiceServer},
    *,
};

/// Embodied-FL gRPC service — P2P federated learning.
///
/// Each node handles its own FedAvg aggregation locally.
/// No central coordinator — all nodes are equal peers.
#[derive(Clone)]
pub struct EmbodiedFlService {
    task_registry: Arc<TaskRegistry>,
    fed_server: Arc<FedServer>,
    contribution: Arc<ContributionTracker>,
    vector_db: Arc<std::sync::RwLock<VectorDb>>,
    audit: Arc<AuditChain>,
}

impl EmbodiedFlService {
    pub fn new(
        task_registry: Arc<TaskRegistry>,
        fed_server: Arc<FedServer>,
        contribution: Arc<ContributionTracker>,
        vector_db: Arc<std::sync::RwLock<VectorDb>>,
        audit: Arc<AuditChain>,
    ) -> Self {
        Self {
            task_registry,
            fed_server,
            contribution,
            vector_db,
            audit,
        }
    }

    pub fn into_federated_server(self) -> FederatedServiceServer<Self> {
        FederatedServiceServer::new(self)
            .max_decoding_message_size(64 * 1024 * 1024)
    }

    pub fn into_task_registry_server(self) -> TaskRegistryServer<Self> {
        TaskRegistryServer::new(self)
    }

    pub fn into_contribution_server(self) -> ContributionServiceServer<Self> {
        ContributionServiceServer::new(self)
    }
}

// ============================================================
// FederatedService — core FL training loop
// ============================================================

#[tonic::async_trait]
impl FederatedService for EmbodiedFlService {
    async fn register_client(
        &self,
        req: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        let r = req.into_inner();
        info!("Register client: {} ({})", r.client_id, r.client_name);

        self.fed_server
            .register_client(&r.client_id, &r.client_name, &r.task_type)
            .map_err(|e| Status::internal(e.to_string()))?;

        self.audit
            .append("client_register", &r.client_id, &format!("name={} task={}", r.client_name, r.task_type))
            .map_err(|e| Status::internal(e.to_string()))?;

        let round = self.fed_server.get_global_model()
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(RegisterResponse {
            success: true,
            message: format!("Client {} registered", r.client_id),
            round_id: round.round_num as i64,
            global_model_version: round.params_hash,
        }))
    }

    async fn unregister_client(
        &self,
        req: Request<UnregisterRequest>,
    ) -> Result<Response<UnregisterResponse>, Status> {
        let r = req.into_inner();
        info!("Unregister client: {}", r.client_id);

        self.audit
            .append("client_unregister", &r.client_id, "unregistered")
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(UnregisterResponse { success: true }))
    }

    async fn report_task(
        &self,
        req: Request<TaskReport>,
    ) -> Result<Response<TaskMatchResult>, Status> {
        let r = req.into_inner();
        info!("Report task from {}: type={}", r.client_id, r.task_type);

        // Insert task vector into vector_db for similarity search
        if !r.task_embedding.is_empty() {
            let mut vdb = self.vector_db.write().unwrap();
            let v: Vec<f32> = r.task_embedding.iter().map(|v| *v as f32).collect();
            let mut meta = std::collections::HashMap::new();
            meta.insert("client_id".to_string(), r.client_id.clone());
            meta.insert("task_type".to_string(), r.task_type.clone());
            let task_id = format!("{}_{}", r.client_id, r.task_type);
            let _ = vdb.insert(&task_id, &v, Some(meta));
        }

        // Search for similar tasks
        let similar: Vec<TaskSimilarity> = if !r.task_embedding.is_empty() {
            let vdb = self.vector_db.read().unwrap();
            let query: Vec<f32> = r.task_embedding.iter().map(|v| *v as f32).collect();
            match vdb.search(&query, 5) {
                Ok(results) => results
                    .into_iter()
                    .map(|sr| TaskSimilarity {
                        client_id: sr.meta.get("client_id").cloned().unwrap_or_default(),
                        task_type: sr.meta.get("task_type").cloned().unwrap_or_default(),
                        similarity: 1.0 - sr.distance,
                        model_compatible: "full".to_string(),
                    })
                    .collect(),
                Err(_) => vec![],
            }
        } else {
            vec![]
        };

        Ok(Response::new(TaskMatchResult {
            matched: !similar.is_empty(),
            similar_tasks: similar,
            aggregation_weight: 1.0,
        }))
    }

    async fn get_global_model(
        &self,
        req: Request<ModelRequest>,
    ) -> Result<Response<ModelResponse>, Status> {
        let r = req.into_inner();
        info!("Get global model: client={} type={}", r.client_id, r.model_type);

        // P2P: Get model bytes from local FedAvg engine
        let model_bytes = self.fed_server.get_global_model_bytes().await
            .map_err(|e| Status::internal(e.to_string()))?;

        let model = self.fed_server.get_global_model()
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(ModelResponse {
            success: true,
            model_weights: model_bytes,
            version: model.params_hash,
            round_id: model.round_num as i64,
            num_contributors: model.num_participants_last_round as i32,
            contributor_weights: std::collections::HashMap::new(),
        }))
    }

    async fn upload_update(
        &self,
        req: Request<UpdateRequest>,
    ) -> Result<Response<UpdateResponse>, Status> {
        let r = req.into_inner();
        info!(
            "Upload update from {}: {} samples, loss={:.4}",
            r.client_id, r.num_local_samples, r.local_loss
        );

        // P2P: Submit update to local FedAvg engine (single source of truth)
        let local_result = self.fed_server
            .submit_update_local(
                &r.client_id,
                self.fed_server.current_round_num() as u32,
                &r.local_weights,
                r.num_local_samples as i64,
                r.local_loss as f32,
            )
            .await;

        if let Err(e) = &local_result {
            tracing::warn!("Local FedAvg submit failed (non-fatal): {}", e);
        }

        // Record contribution
        let loss_improvement = 0.0_f64; // Computed after local aggregation
        let score = ContributionTracker::calculate_score(
            r.num_local_samples,
            loss_improvement,
            0.5, // Default task diversity
        );

        let record = crate::contribution_tracker::ContributionRecord {
            id: 0,
            client_id: r.client_id.clone(),
            round_id: 0,
            task_type: r.model_type.clone(),
            num_samples: r.num_local_samples,
            loss_before: 0.0,
            loss_after: r.local_loss as f64,
            loss_improvement,
            contribution_score: score,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        self.contribution
            .record(&record)
            .map_err(|e| Status::internal(e.to_string()))?;

        // Audit
        self.audit
            .append(
                "model_update",
                &r.client_id,
                &format!("samples={} loss={:.4} score={:.4}", r.num_local_samples, r.local_loss, score),
            )
            .map_err(|e| Status::internal(e.to_string()))?;

        let total = self.contribution.leaderboard(1, None)
            .map(|v| v.first().map(|s| s.total_contribution).unwrap_or(0.0))
            .unwrap_or(0.0);

        Ok(Response::new(UpdateResponse {
            accepted: true,
            message: format!("Update received from {}", r.client_id),
            contribution_score: score as f32,
            total_contribution: total as f32,
        }))
    }

    async fn get_round_status(
        &self,
        req: Request<RoundStatusRequest>,
    ) -> Result<Response<RoundStatusResponse>, Status> {
        let r = req.into_inner();
        info!("Get round status: round_id={}", r.round_id);

        let round_id = if r.round_id > 0 {
            r.round_id
        } else {
            self.fed_server.get_global_model()
                .map(|m| m.round_num as i64)
                .unwrap_or(0)
        };

        let round = self.fed_server.get_round(round_id)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(RoundStatusResponse {
            round_id: round.round_id,
            status: round.status,
            total_clients: round.participants.len() as i32,
            reported_clients: round.updates_received.len() as i32,
            global_loss: round.global_loss as f32,
            global_accuracy: 0.0,
            started_at: 0,
            completed_at: 0,
        }))
    }

    async fn trigger_aggregation(
        &self,
        req: Request<AggregationRequest>,
    ) -> Result<Response<AggregationResponse>, Status> {
        let r = req.into_inner();
        info!("Trigger aggregation: requester={} strategy={}", r.requester_id, r.strategy);

        // P2P: Local FedAvg aggregation (no central coordinator)
        match self.fed_server.trigger_aggregation_local(&self.audit).await {
            Ok(result) => {
                info!("Local FedAvg succeeded: round={} clients={}", result.round, result.num_clients);
                self.audit
                    .append("aggregation", "local", &format!("round={} clients={} strategy={}", result.round, result.num_clients, result.strategy))
                    .map_err(|e| Status::internal(e.to_string()))?;

                Ok(Response::new(AggregationResponse {
                    triggered: true,
                    new_round_id: result.round as i64,
                    new_model_version: format!("p2p_r{}", result.round),
                    num_updates_aggregated: result.num_clients,
                    new_global_loss: 0.0,
                }))
            }
            Err(e) => {
                info!("Local FedAvg failed ({}), no updates to aggregate", e);
                Err(Status::internal(e.to_string()))
            }
        }
    }
}

// ============================================================
// TaskRegistryService — task management
// ============================================================

#[tonic::async_trait]
impl TaskRegistryTrait for EmbodiedFlService {
    async fn register_task(
        &self,
        req: Request<NewTaskRequest>,
    ) -> Result<Response<TaskInfo>, Status> {
        let r = req.into_inner();
        info!("Register task: {} for client {}", r.task_id, r.client_id);

        let task = crate::task_registry::Task {
            task_id: r.task_id.clone(),
            client_id: r.client_id.clone(),
            task_type: r.task_type.clone(),
            description: r.description.clone(),
            status: "active".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            rounds_participated: 0,
            total_contribution: 0.0,
            config_json: serde_json::to_string(&r.config).unwrap_or_default(),
        };

        self.task_registry
            .register(&task)
            .map_err(|e| Status::internal(e.to_string()))?;

        // Insert task embedding into vector_db
        if !r.task_embedding.is_empty() {
            let mut vdb = self.vector_db.write().unwrap();
            let v: Vec<f32> = r.task_embedding.iter().map(|v| *v as f32).collect();
            let mut meta = std::collections::HashMap::new();
            meta.insert("client_id".to_string(), r.client_id.clone());
            meta.insert("task_type".to_string(), r.task_type.clone());
            let _ = vdb.insert(&r.task_id, &v, Some(meta));
        }

        self.audit
            .append("task_register", &r.client_id, &format!("task={} type={}", r.task_id, r.task_type))
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(TaskInfo {
            task_id: task.task_id,
            client_id: task.client_id,
            task_type: task.task_type,
            description: task.description,
            status: task.status,
            created_at: 0,
            updated_at: 0,
            rounds_participated: 0,
            total_contribution: 0.0,
            config: r.config,
        }))
    }

    async fn find_similar_tasks(
        &self,
        req: Request<SimilarTaskRequest>,
    ) -> Result<Response<SimilarTaskResponse>, Status> {
        let r = req.into_inner();
        let top_k = if r.top_k > 0 { r.top_k as usize } else { 5 };

        let vdb = self.vector_db.read().unwrap();
        let query: Vec<f32> = r.query_embedding.iter().map(|v| *v as f32).collect();

        let matches: Vec<TaskMatch> = vdb
            .search(&query, top_k)
            .map_err(|e| Status::internal(e.to_string()))?
            .into_iter()
            .map(|sr| TaskMatch {
                task_id: sr.id.clone(),
                client_id: sr.meta.get("client_id").cloned().unwrap_or_default(),
                task_type: sr.meta.get("task_type").cloned().unwrap_or_default(),
                similarity: 1.0 - sr.distance,
                model_compatible: "full".to_string(),
            })
            .collect();

        Ok(Response::new(SimilarTaskResponse { matches }))
    }

    async fn list_tasks(
        &self,
        req: Request<ListTasksRequest>,
    ) -> Result<Response<ListTasksResponse>, Status> {
        let r = req.into_inner();
        let task_type = if r.task_type.is_empty() { None } else { Some(&r.task_type[..]) };
        let status = if r.status.is_empty() { None } else { Some(&r.status[..]) };

        let tasks = self.task_registry
            .list(task_type, status)
            .map_err(|e| Status::internal(e.to_string()))?;

        let total = tasks.len() as i32;
        let task_infos: Vec<TaskInfo> = tasks
            .into_iter()
            .map(|t| TaskInfo {
                task_id: t.task_id,
                client_id: t.client_id,
                task_type: t.task_type,
                description: t.description,
                status: t.status,
                created_at: 0,
                updated_at: 0,
                rounds_participated: t.rounds_participated,
                total_contribution: t.total_contribution as f32,
                config: serde_json::from_str(&t.config_json).unwrap_or_default(),
            })
            .collect();

        Ok(Response::new(ListTasksResponse {
            tasks: task_infos,
            total,
        }))
    }

    async fn update_task_status(
        &self,
        req: Request<UpdateTaskStatusRequest>,
    ) -> Result<Response<TaskInfo>, Status> {
        let r = req.into_inner();
        info!("Update task status: {} -> {}", r.task_id, r.status);

        self.task_registry
            .update_status(&r.task_id, &r.status)
            .map_err(|e| Status::internal(e.to_string()))?;

        let task = self.task_registry
            .get(&r.task_id)
            .map_err(|e| Status::internal(e.to_string()))?
            .ok_or_else(|| Status::not_found("Task not found"))?;

        Ok(Response::new(TaskInfo {
            task_id: task.task_id,
            client_id: task.client_id,
            task_type: task.task_type,
            description: task.description,
            status: task.status,
            created_at: 0,
            updated_at: 0,
            rounds_participated: task.rounds_participated,
            total_contribution: task.total_contribution as f32,
            config: serde_json::from_str(&task.config_json).unwrap_or_default(),
        }))
    }
}

// ============================================================
// ContributionService — contribution tracking & audit
// ============================================================

#[tonic::async_trait]
impl ContributionService for EmbodiedFlService {
    async fn get_contributions(
        &self,
        req: Request<ContributionQuery>,
    ) -> Result<Response<ContributionResponse>, Status> {
        let r = req.into_inner();
        let limit = if r.limit > 0 { r.limit } else { 10 };

        let records = self.contribution
            .client_history(&r.client_id, limit)
            .map_err(|e| Status::internal(e.to_string()))?;

        let total_score: f32 = records.iter()
            .map(|r| r.contribution_score as f32)
            .sum();

        let entries: Vec<ContributionEntry> = records
            .into_iter()
            .map(|r| ContributionEntry {
                client_id: r.client_id,
                round_id: r.round_id,
                score: r.contribution_score as f32,
                num_samples: r.num_samples,
                loss_improvement: r.loss_improvement as f32,
                timestamp: r.timestamp,
                audit_hash: String::new(),
            })
            .collect();

        Ok(Response::new(ContributionResponse {
            entries,
            total_score,
        }))
    }

    async fn get_leaderboard(
        &self,
        req: Request<LeaderboardRequest>,
    ) -> Result<Response<LeaderboardResponse>, Status> {
        let r = req.into_inner();
        let top_k = if r.top_k > 0 { r.top_k } else { 10 };
        let task_type = if r.task_type.is_empty() { None } else { Some(&r.task_type[..]) };

        let entries = self.contribution
            .leaderboard(top_k, task_type)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(LeaderboardResponse {
            entries: entries
                .into_iter()
                .map(|s| LeaderboardEntry {
                    client_id: s.client_id,
                    client_name: s.client_name,
                    total_contribution: s.total_contribution as f32,
                    rounds_participated: s.rounds_participated,
                    total_samples: s.total_samples,
                    avg_loss_improvement: s.avg_loss_improvement as f32,
                })
                .collect(),
        }))
    }

    async fn verify_audit_chain(
        &self,
        _req: Request<VerifyRequest>,
    ) -> Result<Response<VerifyResponse>, Status> {
        let (valid, length, hash) = self.audit
            .verify_chain()
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(VerifyResponse {
            valid,
            chain_length: length,
            latest_hash: hash,
        }))
    }
}
