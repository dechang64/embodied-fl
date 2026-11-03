//! P2P Network Layer — libp2p-based peer discovery and model gossip.
//!
//! Architecture:
//!   ┌─────────────────────────────────────────┐
//!   │           P2pNetwork (libp2p)           │
//!   │                                         │
//!   │  mDNS ──→ LAN peer discovery            │
//!   │  Gossipsub ──→ Model update broadcast   │
//!   │  Identify ──→ Peer capability exchange   │
//!   │  Ping ──→ Keepalive                     │
//!   │                                         │
//!   │  No central server. No port forwarding.  │
//!   │  Plug and play — like 迅雷.              │
//!   └─────────────────────────────────────────┘
//!
//! Message flow:
//!   1. Local client → gRPC UploadUpdate → FedServer + Gossipsub broadcast
//!   2. Remote peer → Gossipsub receive → FedServer (pending_updates)
//!   3. TriggerAggregation → local FedAvg → Gossipsub broadcast global model
//!   4. Remote peer → Gossipsub receive → merge with local model (weighted avg)

use serde::{Serialize, Deserialize};
use tokio::sync::mpsc;
use tracing::{info, warn};

use libp2p::{
    Swarm, SwarmBuilder, PeerId,
    gossipsub::{self, Behaviour as Gossipsub, ConfigBuilder as GossipsubConfigBuilder, Event as GossipsubEvent, MessageAuthenticity, ValidationMode, Topic},
    mdns,
    identify::{self, Behaviour as Identify, Config as IdentifyConfig},
    ping,
    noise, yamux, tcp,
    swarm::NetworkBehaviour,
    futures::StreamExt,
    swarm::SwarmEvent,
};

// ════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════

const TOPIC_UPDATES: &str = "embodied-fl/updates/v1";
const TOPIC_MODEL: &str = "embodied-fl/model/v1";

// ════════════════════════════════════════════════════════════════
// Message types (serialized as JSON for gossipsub)
// ════════════════════════════════════════════════════════════════

/// Model update broadcast via gossipsub.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelUpdateMessage {
    pub client_id: String,
    pub round: u32,
    pub num_samples: i64,
    pub train_loss: f32,
    pub params: Vec<f32>,
}

/// Global model sync via gossipsub.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GlobalModelMessage {
    pub round: u32,
    pub params: Vec<f32>,
    pub num_clients: i32,
    pub source: String,
}

// ════════════════════════════════════════════════════════════════
// Channel types (communication between P2P task and main app)
// ════════════════════════════════════════════════════════════════

/// Commands from main app → P2P task.
pub enum P2pCommand {
    BroadcastUpdate(ModelUpdateMessage),
    BroadcastModel(GlobalModelMessage),
}

/// Events from P2P task → main app.
pub enum P2pEvent {
    PeerDiscovered { peer_id: String, addr: String },
    PeerExpired { peer_id: String },
    UpdateReceived(ModelUpdateMessage),
    ModelReceived(GlobalModelMessage),
}

// ════════════════════════════════════════════════════════════════
// Network behaviour (combines all libp2p protocols)
// ════════════════════════════════════════════════════════════════

#[derive(NetworkBehaviour)]
struct P2pBehaviour {
    gossipsub: Gossipsub,
    mdns: mdns::tokio::Behaviour,
    identify: Identify,
    ping: ping::Behaviour,
}

// ════════════════════════════════════════════════════════════════
// P2P Network — entry point
// ════════════════════════════════════════════════════════════════

pub struct P2pNetwork;

impl P2pNetwork {
    /// Start the P2P network.
    ///
    /// Returns:
    /// - `cmd_tx`: Channel to send commands (broadcast updates/models)
    /// - `peer_id`: This node's libp2p PeerId
    ///
    /// The event loop runs in a background tokio task.
    /// Events are sent via `event_tx` (provided by caller).
    pub async fn start(
        event_tx: mpsc::UnboundedSender<P2pEvent>,
    ) -> anyhow::Result<(mpsc::UnboundedSender<P2pCommand>, PeerId)> {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();

        let mut swarm = SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|key| {
                // Gossipsub config: allow large messages for model params
                let gossipsub_config = GossipsubConfigBuilder::default()
                    .validation_mode(ValidationMode::None)
                    .max_transmit_size(4 * 1024 * 1024) // 4MB
                    .build()
                    .map_err(|e| -> anyhow::Error { anyhow::anyhow!("gossipsub config: {}", e) })?;

                let gossipsub = Gossipsub::new(
                    MessageAuthenticity::Signed(key.clone()),
                    gossipsub_config,
                ).map_err(|e| -> anyhow::Error { anyhow::anyhow!("gossipsub init: {}", e) })?;

                Ok(P2pBehaviour {
                    gossipsub,
                    mdns: mdns::tokio::Behaviour::new(
                        mdns::Config::default(),
                        key.public().to_peer_id(),
                    )?,
                    identify: Identify::new(IdentifyConfig::new(
                        "embodied-fl/1.0".to_string(),
                        key.public(),
                    )),
                    ping: ping::Behaviour::default(),
                })
            })?
            .build();

        // Subscribe to topics
        let topic_updates = gossipsub::IdentTopic::new(TOPIC_UPDATES);
        let topic_model = gossipsub::IdentTopic::new(TOPIC_MODEL);
        swarm.behaviour_mut().gossipsub.subscribe(&topic_updates)?;
        swarm.behaviour_mut().gossipsub.subscribe(&topic_model)?;

        // Listen on all interfaces, random port (zero-config)
        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        let peer_id = *swarm.local_peer_id();

        // Start event loop in background
        tokio::spawn(async move {
            Self::event_loop(swarm, cmd_rx, event_tx).await;
        });

        info!("P2P network started — PeerId: {}", peer_id);
        info!("mDNS discovery active (LAN plug-and-play)");

        Ok((cmd_tx, peer_id))
    }

    /// Main event loop — handles swarm events and commands.
    async fn event_loop(
        mut swarm: Swarm<P2pBehaviour>,
        mut cmd_rx: mpsc::UnboundedReceiver<P2pCommand>,
        event_tx: mpsc::UnboundedSender<P2pEvent>,
    ) {
        loop {
            tokio::select! {
                // Handle P2P events
                event = swarm.next() => {
                    if let Some(event) = event {
                        Self::handle_swarm_event(event, &mut swarm, &event_tx);
                    }
                }
                // Handle commands from main app
                cmd = cmd_rx.recv() => {
                    if let Some(cmd) = cmd {
                        Self::handle_command(cmd, &mut swarm);
                    } else {
                        info!("P2P command channel closed, shutting down");
                        break;
                    }
                }
            }
        }
    }

    /// Handle swarm events (peer discovery, messages, connections).
    fn handle_swarm_event(
        event: SwarmEvent<P2pBehaviourEvent>,
        swarm: &mut Swarm<P2pBehaviour>,
        event_tx: &mpsc::UnboundedSender<P2pEvent>,
    ) {
        match event {
            SwarmEvent::Behaviour(behaviour_event) => {
                match behaviour_event {
                    P2pBehaviourEvent::Gossipsub(gossipsub_event) => {
                        match gossipsub_event {
                            GossipsubEvent::Message { message, .. } => {
                                let topic = message.topic.as_str();
                                if topic == TOPIC_UPDATES {
                                    match serde_json::from_slice::<ModelUpdateMessage>(&message.data) {
                                        Ok(msg) => {
                                            info!("P2P update received: client={} samples={}", msg.client_id, msg.num_samples);
                                            let _ = event_tx.send(P2pEvent::UpdateReceived(msg));
                                        }
                                        Err(e) => warn!("Failed to decode update message: {}", e),
                                    }
                                } else if topic == TOPIC_MODEL {
                                    match serde_json::from_slice::<GlobalModelMessage>(&message.data) {
                                        Ok(msg) => {
                                            info!("P2P model received: round={} params={}", msg.round, msg.params.len());
                                            let _ = event_tx.send(P2pEvent::ModelReceived(msg));
                                        }
                                        Err(e) => warn!("Failed to decode model message: {}", e),
                                    }
                                }
                            }
                            _ => {} // Ignore subscribe/unsubscribe events
                        }
                    }
                    P2pBehaviourEvent::Mdns(mdns_event) => {
                        match mdns_event {
                            mdns::Event::Discovered(peers) => {
                                for (peer_id, addr) in peers {
                                    info!("mDNS discovered peer: {} at {}", peer_id, addr);
                                    let _ = swarm.dial(addr.clone());
                                    let _ = event_tx.send(P2pEvent::PeerDiscovered {
                                        peer_id: peer_id.to_string(),
                                        addr: addr.to_string(),
                                    });
                                }
                            }
                            mdns::Event::Expired(peers) => {
                                for (peer_id, _) in peers {
                                    let _ = event_tx.send(P2pEvent::PeerExpired {
                                        peer_id: peer_id.to_string(),
                                    });
                                }
                            }
                        }
                    }
                    P2pBehaviourEvent::Identify(identify_event) => {
                        if let identify::Event::Received { peer_id, info, .. } = identify_event {
                            info!("Identified peer {} (agent: {})", peer_id, info.agent_version);
                        }
                    }
                    P2pBehaviourEvent::Ping(ping_event) => {
                        match ping_event.result {
                            Ok(rtt) => {
                                info!("Ping to {}: {}ms", ping_event.peer, rtt.as_millis());
                            }
                            Err(e) => {
                                warn!("Ping to {} failed: {}", ping_event.peer, e);
                            }
                        }
                    }
                }
            }
            SwarmEvent::NewListenAddr { address, .. } => {
                info!("P2P listening on: {}", address);
            }
            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                info!("P2P connected to peer: {}", peer_id);
            }
            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                warn!("P2P disconnected from peer: {}", peer_id);
            }
            _ => {}
        }
    }

    /// Handle commands from main app (broadcast updates/models).
    fn handle_command(cmd: P2pCommand, swarm: &mut Swarm<P2pBehaviour>) {
        match cmd {
            P2pCommand::BroadcastUpdate(msg) => {
                let data = serde_json::to_vec(&msg).unwrap_or_default();
                let topic = gossipsub::IdentTopic::new(TOPIC_UPDATES);
                match swarm.behaviour_mut().gossipsub.publish(topic, data) {
                    Ok(_) => info!("Broadcasted update to peers (client={})", msg.client_id),
                    Err(e) => warn!("Failed to broadcast update: {}", e),
                }
            }
            P2pCommand::BroadcastModel(msg) => {
                let data = serde_json::to_vec(&msg).unwrap_or_default();
                let topic = gossipsub::IdentTopic::new(TOPIC_MODEL);
                match swarm.behaviour_mut().gossipsub.publish(topic, data) {
                    Ok(_) => info!("Broadcasted global model to peers (round={})", msg.round),
                    Err(e) => warn!("Failed to broadcast model: {}", e),
                }
            }
        }
    }
}
