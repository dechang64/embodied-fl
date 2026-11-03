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
mod peer_manager;
mod auth;
mod tls;

use task_registry::TaskRegistry;
use fed_server::FedServer;
use contribution_tracker::ContributionTracker;
use vector_db::VectorDb;
use audit::AuditChain;
use grpc_service::EmbodiedFlService;
use rest_api::ApiState;
use auth::DeviceIdentity;
use tls::ServerTlsConfig;
use peer_manager::PeerManager;

#[derive(Parser)]
#[command(name = "embodied-fl")]
#[command(version = "0.3.0")]
#[command(about = "Embodied-FL — P2P Federated Learning for Embodied Intelligence")]
struct Cli {
    /// gRPC listen port
    #[arg(long, default_value = "50051")]
    grpc_port: u16,

    /// REST listen port
    #[arg(long, default_value = "8080")]
    rest_port: u16,

    /// P2P peer endpoints (comma-separated)
    /// e.g. --peers http://10.7.44.50:50051,http://10.14.64.127:50051
    #[arg(long, env = "EMBODIED_FL_PEERS")]
    peers: Option<String>,

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
    info!("║   Embodied-FL v0.3.0                            ║");
    info!("║   P2P Federated Learning                        ║");
    info!("║   No master/slave — all nodes are equal         ║");
    info!("║   mTLS + Ed25519 + API Key + Gossip             ║");
    info!("╚══════════════════════════════════════════════════╝");
    info!("");

    // - Device Identity (Ed25519) -
    let device_identity = if let Some(ref path) = cli.device_identity_file {
        let path = std::path::Path::new(path);
        auth::load_or_generate_identity(&cli.device_id, Some(path))?
    } else {
        DeviceIdentity::generate(&cli.device_id)
    };

    // - TLS Configuration -
    let tls_config = if cli.gen_tls {
        let cert_dir = std::path::Path::new("./tls-certs");
        tls::generate_dev_certificates(cert_dir)?;
        ServerTlsConfig::mtls(
            cert_dir.join("ca.pem"),
            cert_dir.join("server.pem"),
            cert_dir.join("server-key.pem"),
        )
    } else if let (Some(ca), Some(cert), Some(key)) = (
        &cli.tls_ca, &cli.tls_cert, &cli.tls_key
    ) {
        ServerTlsConfig::mtls(ca, cert, key)
    } else {
        ServerTlsConfig::plaintext()
    };

    // - P2P Peer Manager -
    let peer_manager = Arc::new(PeerManager::new(&cli.device_id));

    // Parse peers from comma-separated list
    if let Some(ref peers_str) = cli.peers {
        for endpoint in peers_str.split(',') {
            let endpoint = endpoint.trim();
            if !endpoint.is_empty() {
                peer_manager.add_peer(endpoint);
                info!("📡 Peer added: {}", endpoint);
            }
        }
    }

    let num_peers = peer_manager.list_peers().len();

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
    ));
    info!("FedServer initialized (P2P local FedAvg)");

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
    info!("P2P mode: {} peer(s) configured", num_peers);
    info!("mTLS = {}, Ed25519 = ACTIVE, API Key = ACTIVE",
          if tls_config.is_tls() { "ACTIVE" } else { "DISABLED" });
    info!("");
    info!("Quick start:");
    info!("  1. Start this node:  embodied-fl --peers http://<peer-ip>:50051");
    info!("  2. On peer node:    embodied-fl --peers http://<this-ip>:50051");
    info!("  3. Open dashboard:   http://localhost:{}", cli.rest_port);

    tokio::select! {
        r = grpc_server => { r?; }
        r = rest_server => { r?; }
    }

    Ok(())
}
