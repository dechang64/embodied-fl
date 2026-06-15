# ── src/vla_service.rs ──
"""
VLA Data Service — gRPC implementation for Vision-Language-Action data management.

Handles episode upload, query, and dataset statistics for VLA federated training.
Episodes are stored in SQLite alongside task registry, with instruction embeddings
indexed in HNSW for task matching.

Bridge to Python:
  Python VLAFLTrainer → gRPC UploadEpisode → Rust stores + indexes
  Rust VLADataService → gRPC QueryEpisodes → Python VLADataset.from_server()
"""

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use rusqlite::params;
use chrono::Utc;
use tracing::info;

use crate::task_registry::TaskRegistry;
use crate::vector_db::VectorDb;
use crate::audit::AuditChain;

/// VLA Episode storage — SQLite-backed episode management.
pub struct VLAEpisodeStore {
    conn: std::sync::Mutex<rusqlite::Connection>,
}

/// Stored episode summary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VLAEpisodeRecord {
    pub episode_id: String,
    pub client_id: String,
    pub robot_type: String,
    pub task_type: String,
    pub num_steps: i32,
    pub avg_reward: f64,
    pub is_success: bool,
    pub instruction: String,
    pub created_at: String,
    pub metadata_json: String,
}

impl VLAEpisodeStore {
    /// Create a new episode store, initializing the SQLite table.
    pub fn new(db_path: &std::path::Path) -> Result<Self> {
        let conn = rusqlite::Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS vla_episodes (
                episode_id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                robot_type TEXT NOT NULL DEFAULT 'unknown',
                task_type TEXT NOT NULL DEFAULT 'custom',
                num_steps INTEGER NOT NULL DEFAULT 0,
                avg_reward REAL NOT NULL DEFAULT 0.0,
                is_success INTEGER NOT NULL DEFAULT 0,
                instruction TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_vla_episodes_client ON vla_episodes(client_id);
            CREATE INDEX IF NOT EXISTS idx_vla_episodes_task ON vla_episodes(task_type);
            CREATE INDEX IF NOT EXISTS idx_vla_episodes_robot ON vla_episodes(robot_type);
            "
        )?;
        Ok(Self { conn: std::sync::Mutex::new(conn) })
    }

    /// Insert a new episode record.
    pub fn insert(
        &self,
        episode_id: &str,
        client_id: &str,
        robot_type: &str,
        task_type: &str,
        num_steps: i32,
        avg_reward: f64,
        is_success: bool,
        instruction: &str,
        metadata_json: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR REPLACE INTO vla_episodes
             (episode_id, client_id, robot_type, task_type, num_steps,
              avg_reward, is_success, instruction, created_at, metadata_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                episode_id, client_id, robot_type, task_type, num_steps,
                avg_reward, is_success as i32, instruction, now, metadata_json,
            ],
        )?;
        Ok(())
    }

    /// Query episodes with optional filters.
    pub fn query(
        &self,
        client_id: Option<&str>,
        task_type: Option<&str>,
        robot_type: Option<&str>,
        success_only: bool,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<VLAEpisodeRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut sql = String::from(
            "SELECT episode_id, client_id, robot_type, task_type, num_steps,
                    avg_reward, is_success, instruction, created_at, metadata_json
             FROM vla_episodes WHERE 1=1"
        );
        if let Some(cid) = client_id {
            sql.push_str(&format!(" AND client_id = '{}'", cid));
        }
        if let Some(tt) = task_type {
            sql.push_str(&format!(" AND task_type = '{}'", tt));
        }
        if let Some(rt) = robot_type {
            sql.push_str(&format!(" AND robot_type = '{}'", rt));
        }
        if success_only {
            sql.push_str(" AND is_success = 1");
        }
        sql.push_str(" ORDER BY created_at DESC LIMIT ? OFFSET ?");

        let mut stmt = conn.prepare(&sql)?;
        let rows: Vec<VLAEpisodeRecord> = stmt
            .query_map(params![limit, offset], |row| {
                Ok(VLAEpisodeRecord {
                    episode_id: row.get(0)?,
                    client_id: row.get(1)?,
                    robot_type: row.get(2)?,
                    task_type: row.get(3)?,
                    num_steps: row.get(4)?,
                    avg_reward: row.get(5)?,
                    is_success: row.get::<_, i32>(6)? != 0,
                    instruction: row.get(7)?,
                    created_at: row.get(8)?,
                    metadata_json: row.get(9)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Get dataset statistics.
    pub fn stats(&self, client_id: Option<&str>) -> Result<VLADatasetStats> {
        let conn = self.conn.lock().unwrap();

        let filter = client_id
            .map(|c| format!("WHERE client_id = '{}'", c))
            .unwrap_or_default();

        let total_episodes: i32 = conn.query_row(
            &format!("SELECT COUNT(*) FROM vla_episodes {}", filter),
            [],
            |r| r.get(0),
        )?;

        let total_steps: i32 = conn.query_row(
            &format!("SELECT COALESCE(SUM(num_steps), 0) FROM vla_episodes {}", filter),
            [],
            |r| r.get(0),
        )?;

        let successful: i32 = conn.query_row(
            &format!("SELECT COUNT(*) FROM vla_episodes {} AND is_success = 1",
                     if filter.is_empty() { "WHERE".into() } else { "AND".into() }),
            [],
            |r| r.get(0),
        ).unwrap_or(0);

        let avg_reward: f64 = conn.query_row(
            &format!("SELECT COALESCE(AVG(avg_reward), 0) FROM vla_episodes {}", filter),
            [],
            |r| r.get(0),
        )?;

        // Per-task-type counts
        let mut episodes_by_task: HashMap<String, i32> = HashMap::new();
        let task_sql = format!(
            "SELECT task_type, COUNT(*) FROM vla_episodes {} GROUP BY task_type",
            filter
        );
        let mut stmt = conn.prepare(&task_sql)?;
        let rows: Vec<(String, i32)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        for (tt, count) in rows {
            episodes_by_task.insert(tt, count);
        }

        // Per-robot-type counts
        let mut episodes_by_robot: HashMap<String, i32> = HashMap::new();
        let robot_sql = format!(
            "SELECT robot_type, COUNT(*) FROM vla_episodes {} GROUP BY robot_type",
            filter
        );
        let mut stmt = conn.prepare(&robot_sql)?;
        let rows: Vec<(String, i32)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        for (rt, count) in rows {
            episodes_by_robot.insert(rt, count);
        }

        // Per-client step counts
        let mut steps_by_client: HashMap<String, i32> = HashMap::new();
        let client_sql = "SELECT client_id, SUM(num_steps) FROM vla_episodes GROUP BY client_id";
        let mut stmt = conn.prepare(client_sql)?;
        let rows: Vec<(String, i32)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        for (cid, steps) in rows {
            steps_by_client.insert(cid, steps);
        }

        let avg_steps = if total_episodes > 0 {
            total_steps as f64 / total_episodes as f64
        } else {
            0.0
        };

        Ok(VLADatasetStats {
            total_episodes,
            total_steps,
            successful_episodes: successful,
            avg_reward,
            avg_steps_per_episode: avg_steps,
            episodes_by_task_type: episodes_by_task,
            episodes_by_robot_type: episodes_by_robot,
            steps_by_client,
        })
    }

    /// Count total episodes.
    pub fn count(&self) -> Result<i32> {
        let conn = self.conn.lock().unwrap();
        let count: i32 = conn.query_row(
            "SELECT COUNT(*) FROM vla_episodes",
            [],
            |r| r.get(0),
        )?;
        Ok(count)
    }
}

/// Dataset statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VLADatasetStats {
    pub total_episodes: i32,
    pub total_steps: i32,
    pub successful_episodes: i32,
    pub avg_reward: f64,
    pub avg_steps_per_episode: f64,
    pub episodes_by_task_type: HashMap<String, i32>,
    pub episodes_by_robot_type: HashMap<String, i32>,
    pub steps_by_client: HashMap<String, i32>,
}

/// VLA Data Service — orchestrates episode storage, indexing, and retrieval.
pub struct VLADataService {
    episode_store: VLAEpisodeStore,
    vector_db: Arc<std::sync::RwLock<VectorDb>>,
    audit: Arc<AuditChain>,
}

impl VLADataService {
    pub fn new(
        db_path: &std::path::Path,
        vector_db: Arc<std::sync::RwLock<VectorDb>>,
        audit: Arc<AuditChain>,
    ) -> Result<Self> {
        let episode_store = VLAEpisodeStore::new(db_path)?;
        Ok(Self { episode_store, vector_db, audit })
    }

    /// Upload a single episode.
    pub fn upload_episode(
        &self,
        episode_id: &str,
        client_id: &str,
        robot_type: &str,
        task_type: &str,
        num_steps: i32,
        avg_reward: f64,
        is_success: bool,
        instruction: &str,
        instruction_embedding: Option<&[f32]>,
        metadata: HashMap<String, String>,
    ) -> Result<String> {
        // Store episode record
        let metadata_json = serde_json::to_string(&metadata)?;
        self.episode_store.insert(
            episode_id, client_id, robot_type, task_type,
            num_steps, avg_reward, is_success, instruction, &metadata_json,
        )?;

        // Index instruction embedding in HNSW
        if let Some(embedding) = instruction_embedding {
            let mut vdb = self.vector_db.write().unwrap();
            let mut meta = HashMap::new();
            meta.insert("type".to_string(), "instruction".to_string());
            meta.insert("episode_id".to_string(), episode_id.to_string());
            meta.insert("client_id".to_string(), client_id.to_string());
            meta.insert("task_type".to_string(), task_type.to_string());
            let vec_id = format!("inst_{}", episode_id);
            vdb.insert(&vec_id, embedding, Some(meta))?;
        }

        // Audit
        self.audit.append("vla_episode_upload", &format!(
            "episode={} client={} task={} steps={} reward={:.4} success={}",
            episode_id, client_id, task_type, num_steps, avg_reward, is_success,
        ))?;

        info!(
            "VLA episode uploaded: {} from {} ({} steps, reward={:.4})",
            episode_id, client_id, num_steps, avg_reward,
        );

        Ok(episode_id.to_string())
    }

    /// Query episodes.
    pub fn query_episodes(
        &self,
        client_id: Option<&str>,
        task_type: Option<&str>,
        robot_type: Option<&str>,
        success_only: bool,
        limit: i32,
        offset: i32,
    ) -> Result<(Vec<VLAEpisodeRecord>, i32)> {
        let total = self.episode_store.count()?;
        let episodes = self.episode_store.query(
            client_id, task_type, robot_type, success_only, limit, offset,
        )?;
        Ok((episodes, total))
    }

    /// Get dataset statistics.
    pub fn get_stats(&self, client_id: Option<&str>) -> Result<VLADatasetStats> {
        self.episode_store.stats(client_id)
    }
}
