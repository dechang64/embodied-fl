//! FedCtx gRPC Client — connects to FedCtx (unified-fl-backend) for real FL aggregation.
//!
//! This module bridges embodied-fl's node-side services to FedCtx's FlEngine,
//! which performs real FedAvg/FedProx/EWA parameter aggregation.

use std::sync::Arc;
use anyhow::Result;
use tonic::transport::Channel;
use tracing::{info, warn};

pub mod fedctx_proto {
    tonic::include_proto!("fedctx");
}

use fedctx_proto::{
    federated_learning_client::FederatedLearningClient,
    SubmitUpdateRequest, ModelUpdateProto,
    GetModelRequest, AggregateRequest, ListRoundsRequest,
};

/// Configuration for connecting to FedCtx coordinator.
#[derive(Clone)]
pub struct FedCtxConfig {
    pub endpoint: String,
    pub tls_ca_cert: Option<String>,
    pub tls_client_cert: Option<String>,
    pub tls_client_key: Option<String>,
    pub api_key: Option<String>,
}

impl Default for FedCtxConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:50051".to_string(),
            tls_ca_cert: None,
            tls_client_cert: None,
            tls_client_key: None,
            api_key: None,
        }
    }
}

/// FedCtx gRPC client wrapper.
pub struct FedCtxClient {
    inner: FederatedLearningClient<Channel>,
    config: FedCtxConfig,
}

impl FedCtxClient {
    /// Connect to FedCtx coordinator.
    pub async fn connect(config: FedCtxConfig) -> Result<Arc<Self>> {
        let endpoint = config.endpoint.clone();
        let channel = Channel::from_shared(endpoint)?
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(30))
            .connect()
            .await?;

        let client = FederatedLearningClient::new(channel);

        info!("FedCtx client connected to {}", config.endpoint);

        Ok(Arc::new(Self {
            inner: client,
            config,
        }))
    }

    /// Submit a model update to FedCtx for real FedAvg aggregation.
    ///
    /// Phase 1: This replaces the local fake aggregation that only merged metrics.
    pub async fn submit_update(
        &self,
        client_id: &str,
        round: u32,
        parameters: Vec<f32>,
        num_samples: i64,
        train_loss: f32,
    ) -> Result<SubmitUpdateResult> {
        let update = ModelUpdateProto {
            client_id: client_id.to_string(),
            round,
            parameters,
            num_samples,
            train_loss,
            entropy: 0.0,
        };

        let mut req = tonic::Request::new(SubmitUpdateRequest {
            update: Some(update),
        });

        if let Some(ref key) = self.config.api_key {
            req.metadata_mut()
                .insert("x-api-key", key.parse().unwrap());
        }

        let resp = self.inner.clone().submit_update(req).await?;
        let resp = resp.into_inner();

        Ok(SubmitUpdateResult {
            accepted: resp.accepted,
            current_round: resp.current_round,
            updates_received: resp.updates_received,
        })
    }

    /// Trigger aggregation on FedCtx (real FedAvg).
    pub async fn aggregate(&self) -> Result<AggregateResult> {
        let mut req = tonic::Request::new(AggregateRequest {});

        if let Some(ref key) = self.config.api_key {
            req.metadata_mut()
                .insert("x-api-key", key.parse().unwrap());
        }

        let resp = self.inner.clone().aggregate(req).await?;
        let resp = resp.into_inner();

        Ok(AggregateResult {
            success: resp.success,
            round: resp.round,
            num_clients: resp.num_clients,
            strategy: resp.strategy,
        })
    }

    /// Get the current global model from FedCtx.
    ///
    /// Phase 1: This returns real aggregated parameters, not empty bytes.
    pub async fn get_model(&self) -> Result<GetModelResult> {
        let mut req = tonic::Request::new(GetModelRequest {});

        if let Some(ref key) = self.config.api_key {
            req.metadata_mut()
                .insert("x-api-key", key.parse().unwrap());
        }

        let resp = self.inner.clone().get_model(req).await?;
        let resp = resp.into_inner();

        Ok(GetModelResult {
            parameters: resp.parameters,
            round: resp.round,
            num_clients: resp.num_clients,
            strategy: resp.strategy,
        })
    }

    /// List FL rounds from FedCtx.
    pub async fn list_rounds(&self, limit: i32) -> Result<Vec<RoundInfo>> {
        let mut req = tonic::Request::new(ListRoundsRequest { limit });

        if let Some(ref key) = self.config.api_key {
            req.metadata_mut()
                .insert("x-api-key", key.parse().unwrap());
        }

        let resp = self.inner.clone().list_rounds(req).await?;
        let resp = resp.into_inner();

        Ok(resp
            .rounds
            .into_iter()
            .map(|r| RoundInfo {
                round: r.round,
                num_updates: r.num_updates,
                status: r.status,
                client_ids: r.client_ids,
            })
            .collect())
    }

    /// Check if FedCtx is reachable.
    pub async fn health_check(&self) -> bool {
        match self.get_model().await {
            Ok(_) => true,
            Err(e) => {
                warn!("FedCtx health check failed: {}", e);
                false
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubmitUpdateResult {
    pub accepted: bool,
    pub current_round: u32,
    pub updates_received: i32,
}

#[derive(Debug, Clone)]
pub struct GetModelResult {
    pub parameters: Vec<f32>,
    pub round: u32,
    pub num_clients: i32,
    pub strategy: String,
}

#[derive(Debug, Clone)]
pub struct AggregateResult {
    pub success: bool,
    pub round: u32,
    pub num_clients: i32,
    pub strategy: String,
}

#[derive(Debug, Clone)]
pub struct RoundInfo {
    pub round: u32,
    pub num_updates: i32,
    pub status: String,
    pub client_ids: Vec<String>,
}
