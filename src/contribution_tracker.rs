use anyhow::Result;
use rusqlite::{Connection, params};
use std::path::Path;
use std::sync::Mutex;
use chrono::Utc;
use serde::{Serialize, Deserialize};

/// 贡献追踪器 — 量化每个客户端的数据贡献
///
/// 核心设计：
/// 1. 每次模型更新记录贡献（样本数、loss改善、梯度质量）
/// 2. 贡献分数 = f(样本数, loss改善, 数据多样性, 参与频率)
/// 3. 为未来的数据市场提供定价依据
pub struct ContributionTracker {
    conn: Mutex<Connection>,
}

/// 贡献记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionRecord {
    pub id: i64,
    pub client_id: String,
    pub round_id: i64,
    pub task_type: String,
    pub num_samples: i32,
    pub loss_before: f64,
    pub loss_after: f64,
    pub loss_improvement: f64,
    pub contribution_score: f64,
    pub timestamp: String,
}

/// 贡献汇总
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionSummary {
    pub client_id: String,
    pub client_name: String,
    pub task_type: String,
    pub total_contribution: f64,
    pub rounds_participated: i32,
    pub total_samples: i32,
    pub avg_loss_improvement: f64,
    pub best_loss_improvement: f64,
}

impl ContributionTracker {
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;"
        )?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                task_type TEXT NOT NULL,
                num_samples INTEGER NOT NULL,
                loss_before REAL NOT NULL DEFAULT 0.0,
                loss_after REAL NOT NULL DEFAULT 0.0,
                loss_improvement REAL NOT NULL DEFAULT 0.0,
                contribution_score REAL NOT NULL DEFAULT 0.0,
                timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_contrib_client ON contributions(client_id);
            CREATE INDEX IF NOT EXISTS idx_contrib_round ON contributions(round_id);"
        )?;

        Ok(Self { conn: Mutex::new(conn) })
    }

    /// 记录一次贡献
    ///
    /// 贡献分数计算公式：
    /// score = num_samples_norm * 0.3 + loss_improvement_norm * 0.5 + diversity_bonus * 0.2
    pub fn record(&self, record: &ContributionRecord) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO contributions (client_id, round_id, task_type, num_samples, loss_before, loss_after, loss_improvement, contribution_score, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                record.client_id, record.round_id, record.task_type,
                record.num_samples, record.loss_before, record.loss_after,
                record.loss_improvement, record.contribution_score, record.timestamp
            ],
        )?;
        Ok(())
    }

    /// 获取贡献排行榜
    pub fn leaderboard(&self, top_k: i32, task_type: Option<&str>) -> Result<Vec<ContributionSummary>> {
        let conn = self.conn.lock().unwrap();
        let mut sql = String::from(
            "SELECT
                c.client_id,
                COALESCE(cl.client_name, c.client_id) as client_name,
                c.task_type,
                SUM(c.contribution_score) as total_contribution,
                COUNT(DISTINCT c.round_id) as rounds_participated,
                SUM(c.num_samples) as total_samples,
                AVG(c.loss_improvement) as avg_loss_improvement,
                MAX(c.loss_improvement) as best_loss_improvement
             FROM contributions c
             LEFT JOIN clients cl ON c.client_id = cl.client_id"
        );
        if task_type.is_some() {
            sql.push_str(&format!(" WHERE c.task_type = '{}'", task_type.unwrap()));
        }
        sql.push_str(" GROUP BY c.client_id ORDER BY total_contribution DESC LIMIT ?1");

        let mut stmt = conn.prepare(&sql)?;
        let entries = stmt.query_map(params![top_k], |row| {
            Ok(ContributionSummary {
                client_id: row.get(0)?,
                client_name: row.get(1)?,
                task_type: row.get(2)?,
                total_contribution: row.get(3)?,
                rounds_participated: row.get(4)?,
                total_samples: row.get(5)?,
                avg_loss_improvement: row.get(6)?,
                best_loss_improvement: row.get(7)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(entries)
    }

    /// 获取某客户端的贡献历史
    pub fn client_history(&self, client_id: &str, limit: i32) -> Result<Vec<ContributionRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, client_id, round_id, task_type, num_samples, loss_before, loss_after, loss_improvement, contribution_score, timestamp
             FROM contributions WHERE client_id = ?1 ORDER BY round_id DESC LIMIT ?2"
        )?;
        let records = stmt.query_map(params![client_id, limit], |row| {
            Ok(ContributionRecord {
                id: row.get(0)?,
                client_id: row.get(1)?,
                round_id: row.get(2)?,
                task_type: row.get(3)?,
                num_samples: row.get(4)?,
                loss_before: row.get(5)?,
                loss_after: row.get(6)?,
                loss_improvement: row.get(7)?,
                contribution_score: row.get(8)?,
                timestamp: row.get(9)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(records)
    }

    /// 计算贡献分数
    ///
    /// score = samples_weight * 0.3 + improvement_weight * 0.5 + diversity_bonus * 0.2
    pub fn calculate_score(num_samples: i32, loss_improvement: f64, task_diversity: f64) -> f64 {
        // 样本数归一化（假设10000为参考值）
        let samples_norm = (num_samples as f64 / 10000.0).min(1.0);
        // loss改善归一化（假设0.1为参考值）
        let improvement_norm = (loss_improvement / 0.1).min(1.0).max(0.0);
        // 多样性奖励（0-1，由任务匹配度决定）
        let diversity = task_diversity.min(1.0).max(0.0);

        samples_norm * 0.3 + improvement_norm * 0.5 + diversity * 0.2
    }
}
