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
    task_registry_server::{TaskRegistryService, TaskRegistryServiceServer},
    contribution_service_server::{ContributionService, ContributionServiceServer},
    *,
};

/// Embodied-FL 聚合服务
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
        Self { task_registry, fed_server, contribution, vector_db, audit }
    }

    pub fn into_federated_server(self) -> FederatedServiceServer<Self> {
        FederatedServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl FederatedService for EmbodiedFlService {
    async fn register_client(&self, req: Request<RegisterRequest>) -> Result<Response<RegisterResponse>, Status> {
        let r = req.into_inner();
        info!("Register client: {} ({})", r.client_id, r.client_name);

        self.fed_server.register_client(&r.client_id, &r.client_name, &r.task_type)
            .map_err(|e| Status::internal(e.to_string()))?;

        self.audit.append("client_register", &format!("{} registered as {}", r.client_id, r.client_name))
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(RegisterResponse {
            success: true,
            message: format!("Client {} registered", r.client_id),
            current_round: self.fed_server.current_round().unwrap_or(0),
        }))
    }

    async fn unregister_client(&self, req: Request<UnregisterRequest>) -> Result<Response<UnregisterResponse>, Status> {
        let r = req.into_inner();
        info!("Unregister client: {}", r.client_id);

        self.fed_server.set_client_status(&r.client_id, "offline")
            .map_err(|e| Status::internal(e.to_string()))?;

        self.audit.append("client_unregister", &format!("{} unregistered", r.client_id))
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(UnregisterResponse { success: true }))
    }

    async fn report_task(&self, req: Request<TaskReport>) -> Result<Response<TaskMatchResult>, Status> {
        let r = req.into_inner();
        info!("Task report from {}: {} ({})", r.client_id, r.task_type, r.description);

        // 注册任务
        let task = self.task_registry.register(
            &r.client_id, &r.task_type, &r.description, &r.config_json
        ).map_err(|e| Status::internal(e.to_string()))?;

        // HNSW 搜索相似任务
        let vdb = self.vector_db.read().unwrap();
        let similar = if let Some(ref vec) = r.task_vector {
            vdb.search(&vec.iter().map(|v| *v as f32).collect::<Vec<_>>(), 5)
                .unwrap_or_default()
                .into_iter()
                .map(|(id, dist)| SimilarTask { task_id: id, distance: dist, task_type: String::new(), client_id: String::new() })
                .collect()
        } else {
            vec![]
        };
        drop(vdb);

        // 插入任务向量
        if let Some(vec) = r.task_vector {
            let mut vdb = self.vector_db.write().unwrap();
            let v: Vec<f32> = vec.iter().map(|v| *v as f32).collect();
            let mut meta = std::collections::HashMap::new();
            meta.insert("client_id".to_string(), r.client_id.clone());
            meta.insert("task_type".to_string(), r.task_type.clone());
            let _ = vdb.insert(&task.task_id, &v, Some(meta));
        }

        Ok(Response::new(TaskMatchResult {
            task_id: task.task_id,
            similar_tasks: similar,
            recommended_peers: vec![],
        }))
    }

    async fn get_global_model(&self, req: Request<ModelRequest>) -> Result<Response<ModelResponse>, Status> {
        let r = req.into_inner();
        info!("Get global model: round={}", r.round_id);

        let round = self.fed_server.get_round(r.round_id)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(ModelResponse {
            round_id: round.round_id,
            model_bytes: vec![],  // MVP: 模型权重通过文件系统分发
            model_hash: round.model_hash,
            status: round.status,
        }))
    }

    async fn upload_update(&self, req: Request<UpdateRequest>) -> Result<Response<UpdateResponse>, Status> {
        let r = req.into_inner();
        info!("Upload update from {} for round {}", r.client_id, r.round_id);

        // 记录模型更新
        self.fed_server.record_update(
            &r.client_id, r.round_id, &r.task_type,
            r.num_samples as i32, r.local_loss, r.local_accuracy,
            &r.gradient_bytes,
        ).map_err(|e| Status::internal(e.to_string()))?;

        // 记录贡献
        let score = ContributionTracker::calculate_score(
            r.num_samples as i32,
            r.loss_improvement,
            r.task_diversity,
        );
        self.contribution.record(
            &r.client_id, r.round_id, &r.task_type,
            r.num_samples as i32, r.loss_before, r.loss_after,
            r.loss_improvement, score,
        ).map_err(|e| Status::internal(e.to_string()))?;

        // 审计
        self.audit.append("model_update", &format!(
            "client={} round={} samples={} loss={:.4}→{:.4} score={:.4}",
            r.client_id, r.round_id, r.num_samples, r.loss_before, r.loss_after, score
        )).map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(UpdateResponse {
            success: true,
            contribution_score: score,
            message: format!("Update received. Contribution score: {:.4}", score),
        }))
    }

    async fn get_round_status(&self, req: Request<RoundStatusRequest>) -> Result<Response<RoundStatusResponse>, Status> {
        let r = req.into_inner();
        let round = self.fed_server.get_round(r.round_id)
            .map_err(|e| Status::internal(e.to_string()))?;
        let updates_count = self.fed_server.get_round_updates(r.round_id)
            .map(|u| u.len() as i32)
            .unwrap_or(0);

        Ok(Response::new(RoundStatusResponse {
            round_id: round.round_id,
            status: round.status,
            participants: updates_count,
            total_clients: round.total_clients,
            loss: round.loss,
            accuracy: round.accuracy,
        }))
    }

    async fn trigger_aggregation(&self, req: Request<AggregationRequest>) -> Result<Response<AggregationResponse>, Status> {
        let r = req.into_inner();
        info!("Trigger aggregation for round {}", r.round_id);

        let (loss, accuracy) = self.fed_server.aggregate(r.round_id)
            .map_err(|e| Status::internal(e.to_string()))?;

        self.fed_server.complete_round(r.round_id, loss, accuracy)
            .map_err(|e| Status::internal(e.to_string()))?;

        self.audit.append("aggregation", &format!(
            "round={} loss={:.4} accuracy={:.4}", r.round_id, loss, accuracy
        )).map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(AggregationResponse {
            success: true,
            loss, accuracy,
            message: format!("Round {} aggregated: loss={:.4}, accuracy={:.4}", r.round_id, loss, accuracy),
        }))
    }
}

#[tonic::async_trait]
impl TaskRegistryService for EmbodiedFlService {
    async fn register_task(&self, req: Request<NewTaskRequest>) -> Result<Response<TaskInfo>, Status> {
        let r = req.into_inner();
        let task = self.task_registry.register(&r.client_id, &r.task_type, &r.description, &r.config_json)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(TaskInfo {
            task_id: task.task_id, client_id: task.client_id, task_type: task.task_type,
            description: task.description, status: task.status, created_at: task.created_at,
            rounds_participated: task.rounds_participated, total_contribution: task.total_contribution,
        }))
    }

    async fn find_similar_tasks(&self, req: Request<SimilarTaskRequest>) -> Result<Response<SimilarTaskResponse>, Status> {
        let r = req.into_inner();
        let vdb = self.vector_db.read().unwrap();
        let results = vdb.search(&r.task_vector.iter().map(|v| *v as f32).collect::<Vec<_>>(), r.k as usize)
            .unwrap_or_default();
        drop(vdb);

        Ok(Response::new(SimilarTaskResponse {
            similar_tasks: results.into_iter().map(|(id, dist)| SimilarTask {
                task_id: id, distance: dist, task_type: String::new(), client_id: String::new(),
            }).collect(),
        }))
    }

    async fn list_tasks(&self, req: Request<ListTasksRequest>) -> Result<Response<ListTasksResponse>, Status> {
        let r = req.into_inner();
        let tasks = self.task_registry.list(r.task_type.as_deref(), r.status.as_deref(), r.limit.unwrap_or(50))
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(ListTasksResponse {
            tasks: tasks.into_iter().map(|t| TaskInfo {
                task_id: t.task_id, client_id: t.client_id, task_type: t.task_type,
                description: t.description, status: t.status, created_at: t.created_at,
                rounds_participated: t.rounds_participated, total_contribution: t.total_contribution,
            }).collect(),
        }))
    }

    async fn update_task_status(&self, req: Request<UpdateTaskStatusRequest>) -> Result<Response<TaskInfo>, Status> {
        let r = req.into_inner();
        let task = self.task_registry.update_status(&r.task_id, &r.status)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(TaskInfo {
            task_id: task.task_id, client_id: task.client_id, task_type: task.task_type,
            description: task.description, status: task.status, created_at: task.created_at,
            rounds_participated: task.rounds_participated, total_contribution: task.total_contribution,
        }))
    }
}

#[tonic::async_trait]
impl ContributionService for EmbodiedFlService {
    async fn get_contributions(&self, req: Request<ContributionRequest>) -> Result<Response<ContributionResponse>, Status> {
        let r = req.into_inner();
        let records = self.contribution.get_by_client(&r.client_id, r.limit.unwrap_or(20))
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(ContributionResponse {
            records: records.into_iter().map(|c| ContributionRecordMsg {
                id: c.id, client_id: c.client_id, round_id: c.round_id, task_type: c.task_type,
                num_samples: c.num_samples, loss_before: c.loss_before, loss_after: c.loss_after,
                loss_improvement: c.loss_improvement, contribution_score: c.contribution_score,
                timestamp: c.timestamp,
            }).collect(),
        }))
    }

    async fn get_leaderboard(&self, req: Request<LeaderboardRequest>) -> Result<Response<LeaderboardResponse>, Status> {
        let r = req.into_inner();
        let entries = self.contribution.leaderboard(r.top_k.unwrap_or(10), r.task_type.as_deref())
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(LeaderboardResponse {
            entries: entries.into_iter().map(|(cid, cname, score, rounds)| LeaderboardEntry {
                client_id: cid, client_name: cname, total_contribution: score,
                rounds_participated: rounds, total_samples: 0, avg_loss_improvement: 0.0,
            }).collect(),
        }))
    }

    async fn verify(&self, _req: Request<VerifyRequest>) -> Result<Response<VerifyResponse>, Status> {
        let (valid, length, hash) = self.audit.verify_chain()
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(VerifyResponse { valid, chain_length: length, latest_hash: hash }))
    }
}
