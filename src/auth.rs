//! Authentication & Authorization — Phase 2 security layer.
//!
//! Three layers:
//! 1. API Key — simple bearer token for REST/gRPC
//! 2. Ed25519 device identity — each node has a keypair, signs challenges
//! 3. mTLS — mutual TLS for gRPC (configured in tls.rs)

use std::path::Path;
use ed25519_dalek::{SigningKey, VerifyingKey, Signer, Verifier};
use rand::rngs::OsRng;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use tracing::info;

/// Device identity — Ed25519 keypair for node authentication.
#[derive(Clone)]
pub struct DeviceIdentity {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    device_id: String,
}

impl DeviceIdentity {
    /// Generate a new device identity.
    pub fn generate(device_id: &str) -> Self {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();

        info!(
            "Generated device identity: {} (pubkey: {})",
            device_id,
            hex::encode(verifying_key.to_bytes())
        );

        Self {
            signing_key,
            verifying_key,
            device_id: device_id.to_string(),
        }
    }

    /// Load from existing keypair bytes.
    pub fn from_bytes(device_id: &str, secret_bytes: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(secret_bytes);
        let verifying_key = signing_key.verifying_key();

        info!(
            "Loaded device identity: {} (pubkey: {})",
            device_id,
            hex::encode(verifying_key.to_bytes())
        );

        Self {
            signing_key,
            verifying_key,
            device_id: device_id.to_string(),
        }
    }

    /// Get device ID.
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Get public key bytes (32 bytes).
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Get public key as hex string.
    pub fn public_key_hex(&self) -> String {
        hex::encode(self.verifying_key.to_bytes())
    }

    /// Get secret key bytes (32 bytes).
    pub fn secret_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Get verifying key reference.
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }

    /// Sign a message with the device's private key.
    pub fn sign(&self, message: &str) -> [u8; 64] {
        self.signing_key.sign(message.as_bytes()).to_bytes()
    }

    /// Generate an authentication token: device_id:timestamp:signature
    ///
    /// The signature covers "device_id:timestamp" to prevent replay attacks.
    pub fn generate_auth_token(&self) -> String {
        let timestamp = chrono::Utc::now().timestamp();
        let message = format!("{}:{}", self.device_id, timestamp);
        let signature = self.signing_key.sign(message.as_bytes());
        let sig_hex = hex::encode(signature.to_bytes());
        format!("{}:{}:{}", self.device_id, timestamp, sig_hex)
    }

    /// Verify an authentication token.
    ///
    /// Returns Some(device_id) if valid, None if invalid.
    pub fn verify_auth_token(
        token: &str,
        trusted_keys: &[(String, VerifyingKey)],
    ) -> Option<String> {
        let parts: Vec<&str> = token.splitn(3, ':').collect();
        if parts.len() != 3 {
            return None;
        }

        let device_id = parts[0];
        let timestamp: i64 = parts[1].parse().ok()?;
        let sig_bytes = hex::decode(parts[2]).ok()?;
        if sig_bytes.len() != 64 {
            return None;
        }

        let mut sig_arr = [0u8; 64];
        sig_arr.copy_from_slice(&sig_bytes);

        // Check timestamp freshness (±5 minutes)
        let now = chrono::Utc::now().timestamp();
        if (now - timestamp).abs() > 300 {
            return None;
        }

        // Find trusted key for this device
        let (_, verifying_key) = trusted_keys.iter().find(|(id, _)| id == device_id)?;

        let message = format!("{}:{}", device_id, timestamp);
        let signature = ed25519_dalek::Signature::from_bytes(&sig_arr);

        if verifying_key.verify(message.as_bytes(), &signature).is_ok() {
            Some(device_id.to_string())
        } else {
            None
        }
    }
}

/// Save device identity to a JSON file.
pub fn save_device_identity(identity: &DeviceIdentity, path: &Path) -> anyhow::Result<()> {
    #[derive(Serialize)]
    struct IdentityFile {
        device_id: String,
        secret_key: String,
        public_key: String,
    }

    let file = IdentityFile {
        device_id: identity.device_id().to_string(),
        secret_key: hex::encode(identity.secret_key_bytes()),
        public_key: identity.public_key_hex(),
    };

    let json = serde_json::to_string_pretty(&file)?;
    std::fs::write(path, json)?;
    info!("Device identity saved to {}", path.display());
    Ok(())
}

/// Load device identity from a JSON file.
pub fn load_device_identity(path: &Path) -> anyhow::Result<DeviceIdentity> {
    #[derive(Deserialize)]
    struct IdentityFile {
        device_id: String,
        secret_key: String,
        #[allow(dead_code)]
        public_key: String,
    }

    let json = std::fs::read_to_string(path)?;
    let file: IdentityFile = serde_json::from_str(&json)?;

    let secret_bytes = hex::decode(&file.secret_key)?;
    if secret_bytes.len() != 32 {
        anyhow::bail!("Invalid secret key length: expected 32, got {}", secret_bytes.len());
    }

    let mut arr = [0u8; 32];
    arr.copy_from_slice(&secret_bytes);

    Ok(DeviceIdentity::from_bytes(&file.device_id, &arr))
}

/// Load or generate device identity.
pub fn load_or_generate_identity(
    device_id: &str,
    path: Option<&Path>,
) -> anyhow::Result<DeviceIdentity> {
    if let Some(p) = path {
        if p.exists() {
            return load_device_identity(p);
        }
        let identity = DeviceIdentity::generate(device_id);
        save_device_identity(&identity, p)?;
        return Ok(identity);
    }
    Ok(DeviceIdentity::generate(device_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_identity_sign_verify() {
        let identity = DeviceIdentity::generate("test-device");
        let challenge = "test-challenge-123";
        let signature = identity.sign(challenge);

        let pub_key = identity.public_key_bytes();
        assert!(verify_signature(&pub_key, challenge, &signature));
    }

    #[test]
    fn test_device_identity_wrong_message() {
        let identity = DeviceIdentity::generate("test-device");
        let signature = identity.sign("message-a");

        let pub_key = identity.public_key_bytes();
        assert!(!verify_signature(&pub_key, "message-b", &signature));
    }

    #[test]
    fn test_save_load_identity() {
        let temp = std::env::temp_dir().join("test_device_identity.json");
        let identity = DeviceIdentity::generate("test-persist");

        save_device_identity(&identity, &temp).unwrap();
        let loaded = load_device_identity(&temp).unwrap();

        assert_eq!(identity.device_id(), loaded.device_id());
        assert_eq!(identity.public_key_hex(), loaded.public_key_hex());
        assert_eq!(identity.secret_key_bytes(), loaded.secret_key_bytes());

        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn test_auth_token_roundtrip() {
        let identity = DeviceIdentity::generate("node-1");
        let token = identity.generate_auth_token();

        let verifying_key = identity.verifying_key.clone();
        let trusted = vec![("node-1".to_string(), verifying_key)];

        let verified = DeviceIdentity::verify_auth_token(&token, &trusted);
        assert_eq!(verified, Some("node-1".to_string()));
    }

    #[test]
    fn test_auth_token_wrong_key() {
        let identity = DeviceIdentity::generate("node-1");
        let token = identity.generate_auth_token();

        // Different identity's key
        let other = DeviceIdentity::generate("node-1");
        let verifying_key = other.verifying_key.clone();
        let trusted = vec![("node-1".to_string(), verifying_key)];

        let verified = DeviceIdentity::verify_auth_token(&token, &trusted);
        assert_eq!(verified, None);
    }
}

/// Verify a signature using public key bytes.
pub fn verify_signature(public_key: &[u8; 32], message: &str, signature: &[u8; 64]) -> bool {
    let Ok(verifying_key) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    let sig = ed25519_dalek::Signature::from_bytes(signature);
    verifying_key.verify(message.as_bytes(), &sig).is_ok()
}
