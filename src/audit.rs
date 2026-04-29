use anyhow::Result;
use chrono::Utc;
use rusqlite::Connection;
use sha2::{Sha256, Digest};
use std::path::Path;
use std::sync::Mutex;

/// 区块链审计日志
///
/// 记录每轮联邦学习的客户端贡献（梯度提交、数据量、损失值），
/// 通过 SHA-256 哈希链确保不可篡改。
/// 为数据贡献度量化和未来数据定价提供可信依据。
pub struct AuditChain {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AuditEntry {
    pub index: i64,
    pub timestamp: String,
    pub operation: String,
    pub client_id: String,
    pub details: String,
    pub hash: String,
    pub prev_hash: String,
}

impl AuditChain {
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS audit_log (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 timestamp TEXT NOT NULL,
                 operation TEXT NOT NULL,
                 client_id TEXT NOT NULL DEFAULT '',
                 details TEXT NOT NULL DEFAULT '',
                 hash TEXT NOT NULL,
                 prev_hash TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_audit_client ON audit_log(client_id);
             CREATE INDEX IF NOT EXISTS idx_audit_operation ON audit_log(operation);
             CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);"
        )?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    /// 追加审计记录
    pub fn append(&self, operation: &str, client_id: &str, details: &str) -> Result<AuditEntry> {
        let conn = self.conn.lock().unwrap();

        let prev_hash: String = conn.query_row(
            "SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get(0),
        ).unwrap_or_else(|_| "GENESIS".to_string());

        let timestamp = Utc::now().to_rfc3339();
        let index: i64 = conn.query_row(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM audit_log",
            [],
            |row| row.get(0),
        )?;

        let mut hasher = Sha256::new();
        hasher.update(index.to_le_bytes());
        hasher.update(timestamp.as_bytes());
        hasher.update(operation.as_bytes());
        hasher.update(client_id.as_bytes());
        hasher.update(details.as_bytes());
        hasher.update(prev_hash.as_bytes());
        let hash = hex::encode(hasher.finalize());

        conn.execute(
            "INSERT INTO audit_log (timestamp, operation, client_id, details, hash, prev_hash) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![timestamp, operation, client_id, details, hash, prev_hash],
        )?;

        Ok(AuditEntry { index, timestamp, operation: operation.to_string(), client_id: client_id.to_string(), details: details.to_string(), hash, prev_hash })
    }

    /// 获取最近审计记录
    pub fn get_recent(
        &self, limit: i64, operation_type: Option<&str>) -> Result<Vec<AuditEntry>> {
        let conn = self.conn.lock().unwrap();
        let mut sql = "SELECT id, timestamp, operation, client_id, details, hash, prev_hash FROM audit_log".to_string();
        if operation_type.is_some() {
            sql.push_str(" WHERE operation = ?1");
        }
        sql.push_str(" ORDER BY id DESC LIMIT ?2");

        let mut stmt = conn.prepare(&sql)?;
        let entries = if let Some(op) = operation_type {
            stmt.query_map(rusqlite::params![op, limit], |row| {
                Ok(AuditEntry {
                    index: row.get(0)?,
                    timestamp: row.get(1)?,
                    operation: row.get(2)?,
                    client_id: row.get(3)?,
                    details: row.get(4)?,
                    hash: row.get(5)?,
                    prev_hash: row.get(6)?,
                })
            })?.collect::<Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(rusqlite::params![limit], |row| {
                Ok(AuditEntry {
                    index: row.get(0)?,
                    timestamp: row.get(1)?,
                    operation: row.get(2)?,
                    client_id: row.get(3)?,
                    details: row.get(4)?,
                    hash: row.get(5)?,
                    prev_hash: row.get(6)?,
                })
            })?.collect::<Result<Vec<_>, _>>()?
        };

        entries.reverse();
        Ok(entries)
    }

    /// 获取客户端的审计记录
    pub fn get_by_client(&self, client_id: &str, limit: i64) -> Result<Vec<AuditEntry>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, timestamp, operation, client_id, details, hash, prev_hash FROM audit_log WHERE client_id = ?1 ORDER BY id DESC LIMIT ?2"
        )?;
        let entries = stmt.query_map(rusqlite::params![client_id, limit], |row| {
            Ok(AuditEntry {
                index: row.get(0)?,
                timestamp: row.get(1)?,
                operation: row.get(2)?,
                client_id: row.get(3)?,
                details: row.get(4)?,
                hash: row.get(5)?,
                prev_hash: row.get(6)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;

        Ok(entries.into_iter().rev().collect())
    }

    /// 验证审计链完整性
    pub fn verify_chain(&self) -> Result<(bool, i64, String)> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM audit_log", [], |row| row.get(0))?;

        let mut stmt = conn.prepare(
            "SELECT id, timestamp, operation, client_id, details, hash, prev_hash FROM audit_log ORDER BY id ASC"
        )?;
        let entries: Vec<AuditEntry> = stmt.query_map([], |row| {
            Ok(AuditEntry {
                index: row.get(0)?, timestamp: row.get(1)?, operation: row.get(2)?,
                client_id: row.get(3)?, details: row.get(4)?, hash: row.get(5)?, prev_hash: row.get(6)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;

        let mut prev_hash = "GENESIS".to_string();
        for entry in &entries {
            if entry.prev_hash != prev_hash {
                return Ok((false, count, entry.hash.clone()));
            }
            let mut hasher = Sha256::new();
            hasher.update(entry.index.to_le_bytes());
            hasher.update(entry.timestamp.as_bytes());
            hasher.update(entry.operation.as_bytes());
            hasher.update(entry.client_id.as_bytes());
            hasher.update(entry.details.as_bytes());
            hasher.update(entry.prev_hash.as_bytes());
            let expected = hex::encode(hasher.finalize());
            if entry.hash != expected {
                return Ok((false, count, entry.hash.clone()));
            }
            prev_hash = entry.hash.clone();
        }

        let latest = entries.last().map(|e| e.hash.clone()).unwrap_or_default();
        Ok((true, count, latest))
    }

    pub fn chain_length(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM audit_log", [], |row| row.get(0))?;
        Ok(count)
    }
}
