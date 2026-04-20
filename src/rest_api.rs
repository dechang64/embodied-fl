use std::sync::Arc;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    Router,
    routing::get,
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::task_registry::TaskRegistry;
use crate::fed_server::FedServer;
use crate::contribution_tracker::ContributionTracker;
use crate::vector_db::VectorDb;
use crate::audit::AuditChain;

#[derive(Clone)]
pub struct ApiState {
    pub task_registry: Arc<TaskRegistry>,
    pub fed_server: Arc<FedServer>,
    pub contribution: Arc<ContributionTracker>,
    pub vector_db: Arc<std::sync::RwLock<VectorDb>>,
    pub audit: Arc<AuditChain>,
}

#[derive(Serialize)]
struct ApiError { error: String }

#[derive(Deserialize)]
struct ListQuery {
    task_type: Option<String>,
    status: Option<String>,
    limit: Option<i32>,
}

pub fn create_router(state: ApiState) -> Router {
    Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/stats", get(stats))
        .route("/api/v1/tasks", get(list_tasks))
        .route("/api/v1/tasks/{task_id}", get(get_task))
        .route("/api/v1/clients", get(list_clients))
        .route("/api/v1/clients/{client_id}/contributions", get(get_contributions))
        .route("/api/v1/leaderboard", get(leaderboard))
        .route("/api/v1/rounds/{round_id}", get(get_round))
        .route("/api/v1/audit/logs", get(audit_logs))
        .route("/api/v1/audit/verify", get(audit_verify))
        .with_state(state)
}

async fn health(State(s): State<ApiState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok", "service": "embodied-fl", "version": "0.1.0"}))
}

async fn stats(State(s): State<ApiState>) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let (total_tasks, active_tasks) = s.task_registry.stats().map_err(|e| err(&e))?;
    let (total_clients, online_clients) = s.fed_server.client_stats().map_err(|e| err(&e))?;
    let vdb = s.vector_db.read().unwrap();
    let (valid, chain_len, _) = s.audit.verify_chain().unwrap_or((false, 0, String::new()));
    let current_round = s.fed_server.current_round().unwrap_or(0);

    Ok(Json(serde_json::json!({
        "current_round": current_round,
        "total_clients": total_clients,
        "online_clients": online_clients,
        "total_tasks": total_tasks,
        "active_tasks": active_tasks,
        "total_vectors": vdb.len(),
        "audit_chain_length": chain_len,
        "audit_chain_valid": valid,
    })))
}

async fn list_tasks(State(s): State<ApiState>, Query(q): Query<ListQuery>) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let tasks = s.task_registry.list(q.task_type.as_deref(), q.status.as_deref(), q.limit.unwrap_or(50)).map_err(|e| err(&e))?;
    Ok(Json(serde_json::json!({ "tasks": tasks })))
}

async fn get_task(State(s): State<ApiState>, Path(task_id): Path<String>) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let task = s.task_registry.get(&task_id).map_err(|e| err(&e))?;
    match task {
        Some(t) => Ok(Json(serde_json::json!(t))),
        None => Err((StatusCode::NOT_FOUND, Json(ApiError { error: "Task not found".into() }))),
    }
}

async fn list_clients(State(s): State<ApiState>) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let clients = s.fed_server.list_clients().map_err(|e| err(&e))?;
    Ok(Json(serde_json::json!({ "clients": clients })))
}

async fn get_contributions(State(s): State<ApiState>, Path(client_id): Path<String>) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let records = s.contribution.get_by_client(&client_id, 50).map_err(|e| err(&e))?;
    let summary = s.contribution.summary(&client_id).ok();
    Ok(Json(serde_json::json!({ "records": records, "summary": summary })))
}

async fn leaderboard(State(s): State<ApiState>) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let entries = s.contribution.leaderboard(20, None).map_err(|e| err(&e))?;
    Ok(Json(serde_json::json!({ "leaderboard": entries })))
}

async fn get_round(State(s): State<ApiState>, Path(round_id): Path<String>) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let rid: i64 = round_id.parse().map_err(|_| (StatusCode::BAD_REQUEST, Json(ApiError { error: "Invalid round ID".into() })))?;
    let round = s.fed_server.get_round(rid).map_err(|e| err(&e))?;
    let updates = s.fed_server.get_round_updates(rid).unwrap_or_default();
    Ok(Json(serde_json::json!({ "round": round, "updates": updates })))
}

async fn audit_logs(State(s): State<ApiState>) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let logs = s.audit.get_recent(100, None).map_err(|e| err(&e))?;
    Ok(Json(serde_json::json!({ "logs": logs })))
}

async fn audit_verify(State(s): State<ApiState>) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let (valid, length, hash) = s.audit.verify_chain().map_err(|e| err(&e))?;
    Ok(Json(serde_json::json!({ "valid": valid, "chain_length": length, "latest_hash": hash })))
}

fn err(e: &anyhow::Error) -> (StatusCode, Json<ApiError>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiError { error: e.to_string() }))
}
