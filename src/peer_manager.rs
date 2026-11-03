//! P2P Peer Manager — manages connections to other embodied-fl nodes.
//!
//! Each node is equal: no master/slave distinction. Nodes discover each other
//! via bootstrap peers and propagate model updates via gossip protocol.
//!
//! Architecture:
//!   Node A ←→ Node B ←→ Node C
//!   Each node runs gRPC server + connects to peers as client.
//!   Model updates are gossiped to all connected peers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Information about a peer node.
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub peer_id: String,
    pub endpoint: String,
    pub connected: bool,
    pub last_seen: Option<Instant>,
    pub rounds_participated: u32,
}

/// Peer manager — tracks all known peers and their connection state.
pub struct PeerManager {
    peers: Mutex<HashMap<String, PeerInfo>>,
    self_id: String,
}

impl PeerManager {
    pub fn new(self_id: &str) -> Self {
        Self {
            peers: Mutex::new(HashMap::new()),
            self_id: self_id.to_string(),
        }
    }

    /// Add a peer from bootstrap list.
    pub fn add_peer(&self, endpoint: &str) {
        let peer_id = endpoint_to_id(endpoint);
        let mut peers = self.peers.lock().unwrap();
        peers.entry(peer_id.clone()).or_insert(PeerInfo {
            peer_id,
            endpoint: endpoint.to_string(),
            connected: false,
            last_seen: None,
            rounds_participated: 0,
        });
    }

    /// Mark peer as connected.
    pub fn mark_connected(&self, endpoint: &str) {
        let peer_id = endpoint_to_id(endpoint);
        let mut peers = self.peers.lock().unwrap();
        if let Some(p) = peers.get_mut(&peer_id) {
            p.connected = true;
            p.last_seen = Some(Instant::now());
            info!("Peer connected: {} ({})", p.peer_id, p.endpoint);
        }
    }

    /// Mark peer as disconnected.
    pub fn mark_disconnected(&self, endpoint: &str) {
        let peer_id = endpoint_to_id(endpoint);
        let mut peers = self.peers.lock().unwrap();
        if let Some(p) = peers.get_mut(&peer_id) {
            p.connected = false;
            warn!("Peer disconnected: {} ({})", p.peer_id, p.endpoint);
        }
    }

    /// Get all peer endpoints.
    pub fn get_peer_endpoints(&self) -> Vec<String> {
        self.peers
            .lock()
            .unwrap()
            .values()
            .map(|p| p.endpoint.clone())
            .collect()
    }

    /// Get connected peer endpoints.
    pub fn get_connected_peers(&self) -> Vec<String> {
        self.peers
            .lock()
            .unwrap()
            .values()
            .filter(|p| p.connected)
            .map(|p| p.endpoint.clone())
            .collect()
    }

    /// List all peers (for dashboard).
    pub fn list_peers(&self) -> Vec<PeerInfo> {
        self.peers
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    /// Get self ID.
    pub fn self_id(&self) -> &str {
        &self.self_id
    }

    /// Check if we have any connected peers.
    pub fn has_connected_peers(&self) -> bool {
        self.peers
            .lock()
            .unwrap()
            .values()
            .any(|p| p.connected)
    }
}

/// Derive a peer ID from endpoint string.
fn endpoint_to_id(endpoint: &str) -> String {
    // e.g. "http://10.7.44.50:50051" → "10.7.44.50:50051"
    endpoint
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .to_string()
}
