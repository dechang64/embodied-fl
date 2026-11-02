use std::sync::Arc;
use clap::Parser;
use tracing::info;

mod task_registry;
mod fed_server;
mod contribution_tracker;
mod audit;
mod vector_db;
mod hnsw_index;
mod grpc_service;
mod rest_api;
mod web_dashboard;
mod task_embedding;
mod vla_service;
mod fedctx_client;
mod auth;
mod tls;

use task_registry::TaskRegistry;
use fed_server::FedServer;
use contribution_tracker::ContributionTracker;
use vector_db::VectorDb;
use audit::AuditChain;
use grpc_service::EmbodiedFlService;
use rest_api::ApiState;
use fedctx_client::{FedCtxClient, FedCtxConfig};
use auth::DeviceIdentity;
use tls::ServerTlsConfig;

#[derive(Parser)]
#[command(name = "embodied-fl")]
#[command(version = "0.2.0")]
#[command(about = "Embodied-FL — Federated Learning for Embodied Intelligence")]
struct Cli {
    /// gRPC listen port
    #[arg(long, default_value = "50051")]
    grpc_port: u16,

    /// REST listen port
    #[arg(long, default_value = "8080")]
    rest_port: u16,

    /// FedCtx coordinator endpoint
    #[arg(long, default_value = "http://127.0.0.1:50052")]
    fedctx_endpoint: String,

    /// API key for authentication
    #[arg(long, env = "EMBODIED_FL_API_KEY", default_value = "dev-key")]
    api_key: String,

    /// Device ID for Ed25519 identity
    #[arg(long, env = "EMBODIED_FL_DEVICE_ID", default_value = "embodied-fl-node-1")]
    device_id: String,

    /// Device identity file path (JSON with Ed25519 keypair)
    #[arg(long, env = "EMBODIED_FL_DEVICE_IDENTITY_FILE")]
    device_identity_file: Option<String>,

    /// TLS CA cert path (enables mTLS if set with server cert/key)
    #[arg(long, env = "EMBODIED_FL_TLS_CA")]
    tls_ca: Option<String>,

    /// TLS server cert path
    #[arg(long, env = "EMBODIED_FL_TLS_CERT")]
    tls_cert: Option<String>,

    /// TLS server key path
    #[arg(long, env = "EMBODIED_FL_TLS_KEY")]
    tls_key: Option<String>,

    /// Generate dev TLS certificates on startup
    #[arg(long, default_value = "false")]
    gen_tls: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "embodied_fl=info,tower_http=info".parse().unwrap())
        )
        .init();

    let cli = Cli::parse();

    info!("");
    info!("╔══════════════════════════════════════════════════╗");
    info!("║   Embodied-FL v0.2.0                            ║");
    info!("║   Federated Learning for Embodied Intelligence   ║");
    info!("║   Phase 1: FedCtx delegation                    ║");
    info!("║   Phase 2: mTLS + Ed25519 + API Key             ║");
    info!("╚══════════════════════════════════════════════════╝");
    info!("");

    // - Phase 2: Device Identity (Ed25519) -
    let device_identity = if let Some(ref path) = cli.device_identity_file {
        let path = std::path::Path::new(path);
        auth::load_or_generate_identity(&cli.device_id, Some(path))?
    } else {
        DeviceIdentity::generate(&cli.device_id)
    };

    // - Phase 2: TLS Configuration -
    let tls_config = if cli.gen_tls {
        let cert_dir = std::path::Path::new("./tls-certs");
        tls::generate_dev_certificates(cert_dir)?;
        ServerTlsConfig::mtls(
            cert_dir.join("ca.pem"),
            cert_dir.join("server.pem"),
            cert_dir.join("server-key.pem"),
        )
    } else if let (Some(ca), Some(cert), Some(key)) = (&cli.tls_ca, &cli.tls_cert, &cli.tls_key) {
        ServerTlsConfig::mtls(ca, cert, key)
    } else {
        ServerTlsConfig::plaintext()
    };

    // - Phase 1: Connect to FedCtx coordinator -
    let fedctx_config = FedCtxConfig {
        endpoint: cli.fedctx_endpoint.clone(),
        tls_ca_cert: cli.tls_ca.clone(),
        tls_client_cert: cli.tls_cert.clone(),
        tls_client_key: cli.tls_key.clone(),
        api_key: Some(cli.api_key.clone()),
    };

    let fedctx_client = match FedCtxClient::connect(fedctx_config).await {
        Ok(client) => {
            info!("✅ Connected to FedCtx at {}", cli.fedctx_endpoint);
            if client.health_check().await {
                info!("✅ FedCtx health check passed");
            } else {
                info!("⚠️  FedCtx health check failed — will retry on demand");
            }
            Some(client)
        }
        Err(e) => {
            info!("⚠️  FedCtx not available ({}), running in standalone mode", e);
            None
        }
    };

    // - Initialize local components -
    std::fs::create_dir_all("data")?;

    let task_registry = Arc::new(TaskRegistry::new(std::path::Path::new("data/tasks.db"))?);
    info!("TaskRegistry initialized");

    let audit = Arc::new(AuditChain::new(std::path::Path::new("data/audit.db"))?);
    info!("AuditChain initialized (SHA-256)");

    let fed_server = Arc::new(FedServer::new(
        Arc::clone(&task_registry),
        Arc::clone(&audit),
        Arc::clone(&task_registry),
        fedctx_client.clone(),
    ));
    info!("FedServer initialized (FedCtx delegation: {})",
          if fedctx_client.is_some() { "ACTIVE" } else { "STANDALONE" });

    let contribution = Arc::new(ContributionTracker::new(std::path::Path::new("data/contributions.db"))?);
    info!("ContributionTracker initialized");

    let vector_db = Arc::new(std::sync::RwLock::new(VectorDb::new(32)));
    info!("VectorDB initialized (HNSW, dim=32)");

    // - gRPC Service -
    let grpc_service = EmbodiedFlService::new(
        Arc::clone(&task_registry),
        Arc::clone(&fed_server),
        Arc::clone(&contribution),
        Arc::clone(&vector_db),
        Arc::clone(&audit),
    );

    let api_state = ApiState {
        task_registry: Arc::clone(&task_registry),
        fed_server: Arc::clone(&fed_server),
        contribution: Arc::clone(&contribution),
        vector_db: Arc::clone(&vector_db),
        audit: Arc::clone(&audit),
        api_key: cli.api_key.clone(),
    };

    // - Start servers -
    let grpc_addr = format!("[::]:{}", cli.grpc_port).parse()?;
    let mut server_builder = tonic::transport::Server::builder();

    if let Some(tls) = tls_config.build_tonic()? {
        info!("Applying mTLS to gRPC server");
        server_builder = server_builder.tls_config(tls)?;
    }

    let grpc_server = server_builder
        .add_service(grpc_service.clone().into_federated_server())
        .add_service(grpc_service.clone().into_task_registry_server())
        .add_service(grpc_service.into_contribution_server())
        .serve(grpc_addr);

    // REST + Dashboard
    let rest_app = rest_api::create_router(api_state)
        .merge(web_dashboard::create_dashboard());
    let rest_listener = tokio::net::TcpListener::bind(
        &format!("0.0.0.0:{}", cli.rest_port)
    ).await?;
    let rest_server = axum::serve(rest_listener, rest_app);

    info!("");
    info!("gRPC server ready on 0.0.0.0:{} (TLS: {})", cli.grpc_port, tls_config.is_tls());
    info!("REST server ready on 0.0.0.0:{}", cli.rest_port);
    info!("Web dashboard: http://0.0.0.0:{}", cli.rest_port);
    info!("Device identity: {} (pubkey: {})",
          device_identity.device_id(),
          device_identity.public_key_hex());
    info!("");
    info!("Phase 1: FedCtx delegation = {}",
          if fedctx_client.is_some() { "ACTIVE" } else { "STANDALONE" });
    info!("Phase 2: mTLS = {}, Ed25519 = ACTIVE, API Key = ACTIVE",
          if tls_config.is_tls() { "ACTIVE" } else { "DISABLED" });
    info!("");
    info!("Quick start:");
    info!("  1. Start FedCtx:  cargo run --bin fedctx -- --grpc-port 50052");
    info!("  2. Start node:    cargo run -- --fedctx-endpoint http://127.0.0.1:50052");
    info!("  3. Open dashboard: http://localhost:{}", cli.rest_port);

    tokio::select! {
        r = grpc_server => { r?; }
        r = rest_server => { r?; }
    }

    Ok(())
}
