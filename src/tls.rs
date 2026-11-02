//! TLS configuration — Phase 2 security layer.
//!
//! Provides mTLS (mutual TLS) for gRPC connections.
//! In development mode, falls back to plaintext.

use std::path::PathBuf;
use tonic::transport::{Certificate, Identity, ServerTlsConfig as TonicServerTlsConfig};

/// TLS configuration for the gRPC server.
#[derive(Clone, Debug)]
pub struct ServerTlsConfig {
    pub ca_cert_path: Option<PathBuf>,
    pub server_cert_path: Option<PathBuf>,
    pub server_key_path: Option<PathBuf>,
}

impl ServerTlsConfig {
    /// No TLS — plaintext mode (development only).
    pub fn plaintext() -> Self {
        Self {
            ca_cert_path: None,
            server_cert_path: None,
            server_key_path: None,
        }
    }

    /// mTLS with all certs provided.
    pub fn mtls(
        ca: impl Into<PathBuf>,
        cert: impl Into<PathBuf>,
        key: impl Into<PathBuf>,
    ) -> Self {
        Self {
            ca_cert_path: Some(ca.into()),
            server_cert_path: Some(cert.into()),
            server_key_path: Some(key.into()),
        }
    }

    /// Check if TLS is enabled.
    pub fn is_tls(&self) -> bool {
        self.server_cert_path.is_some() && self.server_key_path.is_some()
    }

    /// Build tonic ServerTlsConfig.
    pub fn build_tonic(&self) -> anyhow::Result<Option<TonicServerTlsConfig>> {
        if !self.is_tls() {
            return Ok(None);
        }

        let cert_path = self.server_cert_path.as_ref().unwrap();
        let key_path = self.server_key_path.as_ref().unwrap();

        let cert = std::fs::read(cert_path)?;
        let key = std::fs::read(key_path)?;
        let identity = Identity::from_pem(cert, key);

        let mut tls = TonicServerTlsConfig::new().identity(identity);

        // If CA cert provided, require client verification (mTLS)
        if let Some(ca_path) = &self.ca_cert_path {
            let ca = std::fs::read(ca_path)?;
            let ca_cert = Certificate::from_pem(ca);
            tls = tls.client_ca_root(ca_cert);
        }

        Ok(Some(tls))
    }
}

/// Generate self-signed CA and server certificates for development.
///
/// Requires `openssl` CLI to be installed.
pub fn generate_dev_certificates(dir: &std::path::Path) -> anyhow::Result<()> {
    use std::process::Command;

    std::fs::create_dir_all(dir)?;

    // Generate CA key and certificate
    Command::new("openssl")
        .args(["genrsa", "-out"])
        .arg(dir.join("ca-key.pem"))
        .arg("2048")
        .output()?;

    Command::new("openssl")
        .args(["req", "-x509", "-new", "-nodes", "-key"])
        .arg(dir.join("ca-key.pem"))
        .args(["-days", "3650", "-out"])
        .arg(dir.join("ca.pem"))
        .args(["-subj", "/CN=embodied-fl-dev-ca"])
        .output()?;

    // Generate server key and CSR
    Command::new("openssl")
        .args(["genrsa", "-out"])
        .arg(dir.join("server-key.pem"))
        .arg("2048")
        .output()?;

    Command::new("openssl")
        .args(["req", "-new", "-key"])
        .arg(dir.join("server-key.pem"))
        .args(["-out", "/tmp/server.csr", "-subj", "/CN=localhost"])
        .output()?;

    // Sign server cert with CA
    Command::new("openssl")
        .args(["x509", "-req", "-in", "/tmp/server.csr", "-CA"])
        .arg(dir.join("ca.pem"))
        .args(["-CAkey"])
        .arg(dir.join("ca-key.pem"))
        .args(["-CAcreateserial", "-out"])
        .arg(dir.join("server.pem"))
        .args(["-days", "365", "-sha256"])
        .output()?;

    // Generate client key and CSR
    Command::new("openssl")
        .args(["genrsa", "-out"])
        .arg(dir.join("client-key.pem"))
        .arg("2048")
        .output()?;

    Command::new("openssl")
        .args(["req", "-new", "-key"])
        .arg(dir.join("client-key.pem"))
        .args(["-out", "/tmp/client.csr", "-subj", "/CN=embodied-fl-node"])
        .output()?;

    // Sign client cert with CA
    Command::new("openssl")
        .args(["x509", "-req", "-in", "/tmp/client.csr", "-CA"])
        .arg(dir.join("ca.pem"))
        .args(["-CAkey"])
        .arg(dir.join("ca-key.pem"))
        .args(["-CAcreateserial", "-out"])
        .arg(dir.join("client.pem"))
        .args(["-days", "365", "-sha256"])
        .output()?;

    // Clean up CSRs
    let _ = std::fs::remove_file("/tmp/server.csr");
    let _ = std::fs::remove_file("/tmp/client.csr");

    tracing::info!("Generated dev TLS certificates in {}", dir.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plaintext_config() {
        let config = ServerTlsConfig::plaintext();
        assert!(!config.is_tls());
    }

    #[test]
    fn test_mtls_config() {
        let config = ServerTlsConfig::mtls("/ca.pem", "/cert.pem", "/key.pem");
        assert!(config.is_tls());
    }
}
