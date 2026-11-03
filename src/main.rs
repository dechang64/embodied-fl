use std::sync::Arc;
use clap::Parser;
use tracing::{info, warn};

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
mod p2p_network;
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
use p2p_network::{P2pNetwork, P2pEvent, P2pCommand, ModelUpdateMessage, GlobalModelMessage};

#[derive(Parser)]
#[command(name = "embodied-fl")]
#[command(version = "0.4.0")]
#[command(about = "Embodied-FL — P2P Federated Learning, plug and play")]
struct Cli {
    /// gRPC listen port
    #[arg(long, default_value = "50051")]
    grpc_port: u16,

    /// REST listen port
    #[arg(long, default_value = "8080")]
    rest_port: u16,

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

    /// Generate dev TLS certificates
    #[arg(long)]
    gen_tls: bool,

    /// Disable P2P networking (standalone mode)
    #[arg(long)]
    no_p2p: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "embodied_fl=info,libp2p=info".parse().unwrap())
        )
        .init();

    let cli = Cli::parse();

    info!("");
    info!("╔══════════════════════════════════════════════════╗");
    info!("║   Embodied-FL v0.4.0                            ║");
    info!("║   P2P Federated Learning — Plug and Play        ║");
    info!("║   No central coordinator. No port forwarding.   ║");
    info!("╚══════════════════════════════════════════════════╝");
    info!("");

    // ── Device Identity (Ed25519) ──
    let device_identity = if let Some(ref path) = cli.device_identity_file {
        let path = std::path::Path::new(path);
        auth::load_or_generate_identity(&cli.device_id, Some(path))?
    } else {
        DeviceIdentity::generate(&cli.device_id)
    };

    // ── TLS Configuration ──
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

    // ── Core components ──
    let task_registry = Arc::new(TaskRegistry::new(std::path::Path::new("tasks.db"))?);
    let audit = Arc::new(AuditChain::new(std::path::Path::new("audit.db"))?);
    let fed_server = Arc::new(FedServer::new(
        task_registry.clone(),
        audit.clone(),
        task_registry.clone(),
    ));
    let contribution = Arc::new(ContributionTracker::new(std::path::Path::new("contributions.db"))?);
    let vector_db = Arc::new(std::sync::RwLock::new(VectorDb::new(32)));

    // ── P2P Network (libp2p) ──
    let p2p_cmd_tx = if !cli.no_p2p {
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();

        match P2pNetwork::start(event_tx).await {
            Ok((cmd_tx, peer_id)) => {
                info!("P2P active — PeerId: {}", peer_id);

                // Spawn event handler: P2P events → FedServer
                let fed_server_clone = fed_server.clone();
                let audit_clone = audit.clone();
                tokio::spawn(async move {
                    while let Some(event) = event_rx.recv().await {
                        match event {
                            P2pEvent::PeerDiscovered { peer_id, addr } => {
                                info!("Peer discovered: {} at {}", peer_id, addr);
                            }
                            P2pEvent::PeerExpired { peer_id } => {
                                warn!("Peer expired: {}", peer_id);
                            }
                            P2pEvent::UpdateReceived(msg) => {
                                info!("Remote update from {}: {} samples, loss={:.4}",
                                      msg.client_id, msg.num_samples, msg.train_loss);
                                // Convert f32 params → bytes and store
                                let mut bytes = Vec::with_capacity(msg.params.len() * 4);
                                for p in &msg.params {
                                    bytes.extend_from_slice(&p.to_le_bytes());
                                }
                                if let Err(e) = fed_server_clone
                                    .submit_update_local(
                                        &msg.client_id,
                                        msg.round,
                                        &bytes,
                                        msg.num_samples,
                                        msg.train_loss,
                                    )
                                    .await
                                {
                                    warn!("Failed to store remote update: {}", e);
                                }
                            }
                            P2pEvent::ModelReceived(msg) => {
                                info!("Remote model sync: round={}, {} params, {} clients",
                                      msg.round, msg.params.len(), msg.num_clients);
                                // Merge: weighted average by num_clients
                                let local = fed_server_clone.get_global_params();
                                if local.is_empty() {
                                    fed_server_clone.set_global_params(msg.params);
                                } else if !msg.params.is_empty() {
                                    let local_weight = 1.0_f32;
                                    let remote_weight = msg.num_clients as f32;
                                    let total = local_weight + remote_weight;
                                    let merged: Vec<f32> = local
                                        .iter()
                                        .zip(msg.params.iter())
                                        .map(|(l, r)| (l * local_weight + r * remote_weight) / total)
                                        .collect();
                                    fed_server_clone.set_global_params(merged);
                                }
                            }
                        }
                    }
                });

                Some(cmd_tx)
            }
            Err(e) => {
                warn!("P2P network failed to start ({}), running in standalone mode", e);
                None
            }
        }
    } else {
        info!("P2P disabled (--no-p2p)");
        None
    };

    // ── gRPC service ──
    let grpc_service = EmbodiedFlService::new(
        task_registry.clone(),
        fed_server.clone(),
        contribution.clone(),
        vector_db.clone(),
        audit.clone(),
        p2p_cmd_tx.clone(),
    );

    let grpc_addr = format!("0.0.0.0:{}", cli.grpc_port).parse()?;
    let mut grpc_builder = tonic::transport::Server::builder();
    let grpc_server = grpc_builder
        .add_service(grpc_service.clone().into_federated_server())
        .add_service(grpc_service.clone().into_task_registry_server())
        .add_service(grpc_service.into_contribution_server())
        .serve(grpc_addr);

    // ── REST + Dashboard ──
    let api_state = ApiState {
        task_registry: task_registry.clone(),
        fed_server: fed_server.clone(),
        contribution: contribution.clone(),
        vector_db: vector_db.clone(),
        audit: audit.clone(),
        api_key: cli.api_key.clone(),
    };
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
    if p2p_cmd_tx.is_some() {
        info!("P2P mode: ACTIVE (mDNS auto-discovery + gossipsub)");
        info!("Just start another node on the same LAN — it connects automatically.");
    } else {
        info!("P2P mode: DISABLED (standalone)");
    }
    info!("mTLS = {}, Ed25519 = ACTIVE, API Key = ACTIVE",
          if tls_config.is_tls() { "ACTIVE" } else { "DISABLED" });
    info!("");
    info!("Dashboard: http://localhost:{}", cli.rest_port);

    tokio::select! {
        r = grpc_server => { r?; }
        r = rest_server => { r?; }
    }

    Ok(())
}
