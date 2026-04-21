use std::sync::Arc;
use tracing::info;
use tracing_subscriber;

mod task_registry;
mod fed_server;
mod contribution_tracker;
mod audit;
mod vector_db;
mod hnsw_index;
mod grpc_service;
mod rest_api;
mod web_dashboard;

use task_registry::TaskRegistry;
use fed_server::FedServer;
use contribution_tracker::ContributionTracker;
use vector_db::VectorDb;
use audit::AuditChain;
use grpc_service::EmbodiedFlService;
use rest_api::ApiState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "embodied_fl=info,tower_http=debug".parse().unwrap())
        )
        .init();

    info!("╔══════════════════════════════════════════════════╗");
    info!("║   Embodied-FL v0.1.0                             ║");
    info!("║   Federated Learning for Embodied Intelligence   ║");
    info!("╚══════════════════════════════════════════════════╝");

    // 初始化各组件
    let task_registry = Arc::new(TaskRegistry::new(std::path::Path::new("data/tasks.db"))?);
    info!("TaskRegistry initialized");

    let fed_server = Arc::new(FedServer::new(std::path::Path::new("data/fed.db"))?);
    info!("FedServer initialized");

    let contribution = Arc::new(ContributionTracker::new(std::path::Path::new("data/contribution.db"))?);
    info!("ContributionTracker initialized");

    let vector_db = Arc::new(std::sync::RwLock::new(VectorDb::new(32)));
    info!("VectorDB initialized (dimension=32)");

    let audit = Arc::new(AuditChain::new(std::path::Path::new("data/audit.db"))?);
    info!("AuditChain initialized");

    audit.append("system_start", "Embodied-FL v0.1.0 started")?;

    // gRPC 服务
    let grpc_service = EmbodiedFlService::new(
        Arc::clone(&task_registry),
        Arc::clone(&fed_server),
        Arc::clone(&contribution),
        Arc::clone(&vector_db),
        Arc::clone(&audit),
    );

    // REST API
    let api_state = ApiState {
        task_registry: Arc::clone(&task_registry),
        fed_server: Arc::clone(&fed_server),
        contribution: Arc::clone(&contribution),
        vector_db: Arc::clone(&vector_db),
        audit: Arc::clone(&audit),
    };

    // gRPC:50051
    let grpc_addr = "[::]:50051".parse()?;
    let grpc_server = tonic::transport::Server::builder()
        .add_service(grpc_service.into_federated_server())
        .serve(grpc_addr);

    // REST + Dashboard:8080
    let rest_app = rest_api::create_router(api_state)
        .merge(web_dashboard::create_dashboard());
    let rest_listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    let rest_server = axum::serve(rest_listener, rest_app);

    info!("gRPC server ready on 0.0.0.0:50051");
    info!("REST server ready on 0.0.0.0:8080");
    info!("Web dashboard: http://0.0.0.0:8080");
    info!("");
    info!("Quick start:");
    info!("  1. cargo run                    # Start server");
    info!("  2. python python/sim/client.py  # Start simulated robot client");
    info!("  3. Open http://localhost:8080   # View dashboard");
    info!("");

    tokio::select! {
        r = grpc_server => { r?; }
        r = rest_server => { r?; }
    }

    Ok(())
}
