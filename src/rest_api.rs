use std::sync::Arc;
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{Json, IntoResponse, Response},
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
    pub api_key: String,
}

#[derive(Serialize)]
struct ApiError { error: String }

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(self)).into_response()
    }
}

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
        .route("/api/v1/clients", get(list_clients))
        .route("/api/v1/contributions/:client_id", get(get_contributions))
        .route("/api/v1/leaderboard", get(leaderboard))
        .route("/api/v1/rounds/:round_id", get(get_round))
        .route("/api/v1/audit/logs", get(audit_logs))
        .route("/api/v1/audit/verify", get(audit_verify))
        .with_state(state)
}

fn check_api_key(headers: &HeaderMap, api_key: &str) -> Result<(), (StatusCode, Json<ApiError>)> {
    let provided = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if provided != api_key {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ApiError { error: "Invalid or missing API key".into() }),
        ));
    }
    Ok(())
}

async fn health(State(s): State<ApiState>) -> Json<serde_json::Value> {
    let (total_tasks, active_tasks) = s.task_registry.stats().unwrap_or((0, 0));
    Json(serde_json::json!({
        "status": "ok",
        "version": "0.2.0",
        "total_tasks": total_tasks,
        "active_tasks": active_tasks,
        "vector_db_size": s.vector_db.read().unwrap().len(),
        "audit_chain_length": s.audit.chain_length().unwrap_or(0),
    }))
}

async fn stats(State(s): State<ApiState>, headers: HeaderMap) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    check_api_key(&headers, &s.api_key)?;
    let (total_tasks, active_tasks) = s.task_registry.stats().map_err(|e| err(&e))?;
    let clients = s.fed_server.list_registered_clients();
    let current_round = s.fed_server.current_round_num();
    Ok(Json(serde_json::json!({
        "tasks": { "total": total_tasks, "active": active_tasks },
        "clients": clients.len(),
        "current_round": current_round,
    })))
}

async fn list_tasks(
    State(s): State<ApiState>,
    headers: HeaderMap,
    Query(q): Query<ListQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    check_api_key(&headers, &s.api_key)?;
    let tasks = s.task_registry
        .list(q.task_type.as_deref(), q.status.as_deref())
        .map_err(|e| err(&e))?;
    Ok(Json(serde_json::json!({ "tasks": tasks, "total": tasks.len() })))
}

async fn list_clients(
    State(s): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    check_api_key(&headers, &s.api_key)?;
    let clients = s.fed_server.list_registered_clients();
    Ok(Json(serde_json::json!({ "clients": clients })))
}

async fn get_contributions(
    State(s): State<ApiState>,
    headers: HeaderMap,
    Path(client_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    check_api_key(&headers, &s.api_key)?;
    let records = s.contribution
        .client_history(&client_id, 100)
        .map_err(|e| err(&e))?;
    Ok(Json(serde_json::json!({ "records": records })))
}

async fn leaderboard(
    State(s): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    check_api_key(&headers, &s.api_key)?;
    let entries = s.contribution.leaderboard(20, None).map_err(|e| err(&e))?;
    Ok(Json(serde_json::json!({ "leaderboard": entries })))
}

async fn get_round(
    State(s): State<ApiState>,
    headers: HeaderMap,
    Path(round_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    check_api_key(&headers, &s.api_key)?;
    let rid: i64 = round_id.parse().map_err(|_| bad("Invalid round ID"))?;
    let round = s.fed_server.get_round(rid).map_err(|e| err(&e))?;
    let updates = s.fed_server.get_round_updates(rid).unwrap_or_default();
    Ok(Json(serde_json::json!({ "round": round, "updates": updates })))
}

async fn audit_logs(
    State(s): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    check_api_key(&headers, &s.api_key)?;
    let logs = s.audit.get_recent(100, None).map_err(|e| err(&e))?;
    Ok(Json(serde_json::json!({ "logs": logs })))
}

async fn audit_verify(
    State(s): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    check_api_key(&headers, &s.api_key)?;
    let (valid, length, hash) = s.audit.verify_chain().map_err(|e| err(&e))?;
    Ok(Json(serde_json::json!({ "valid": valid, "chain_length": length, "latest_hash": hash })))
}

fn err(e: &anyhow::Error) -> (StatusCode, Json<ApiError>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiError { error: e.to_string() }))
}

fn bad(msg: &str) -> (StatusCode, Json<ApiError>) {
    (StatusCode::BAD_REQUEST, Json(ApiError { error: msg.into() }))
}
