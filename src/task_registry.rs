use anyhow::Result;
use rusqlite::{Connection, params};
use std::path::Path;
use std::sync::Mutex;
use chrono::Utc;

/// 任务注册表 — 管理异构具身智能任务
///
/// 核心功能：
/// 1. 注册不同类型的任务（抓取/导航/装配/检测）
/// 2. 为每个任务生成特征向量（用于HNSW相似度匹配）
/// 3. 跟踪任务状态和参与轮次
pub struct TaskRegistry {
    conn: Mutex<Connection>,
}

/// 任务信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Task {
    pub task_id: String,
    pub client_id: String,
    pub task_type: String,
    pub description: String,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
    pub rounds_participated: i32,
    pub total_contribution: f64,
    pub config_json: String,
}

/// 任务类型枚举
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TaskType {
    Grasping,       // 机械臂抓取
    Navigation,     // 移动机器人导航
    Assembly,       // 装配任务
    Inspection,     // 质量检测
    Manipulation,   // 物体操作
    Custom(String), // 自定义任务
}

impl TaskType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "grasping" => TaskType::Grasping,
            "navigation" => TaskType::Navigation,
            "assembly" => TaskType::Assembly,
            "inspection" => TaskType::Inspection,
            "manipulation" => TaskType::Manipulation,
            other => TaskType::Custom(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            TaskType::Grasping => "grasping",
            TaskType::Navigation => "navigation",
            TaskType::Assembly => "assembly",
            TaskType::Inspection => "inspection",
            TaskType::Manipulation => "manipulation",
            TaskType::Custom(s) => s,
        }
    }

    /// 任务类型的 one-hot 编码（用于特征向量）
    pub fn one_hot(&self) -> [f32; 6] {
        match self {
            TaskType::Grasping => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            TaskType::Navigation => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            TaskType::Assembly => [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            TaskType::Inspection => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            TaskType::Manipulation => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            TaskType::Custom(_) => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl TaskRegistry {
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;"
        )?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                rounds_participated INTEGER NOT NULL DEFAULT 0,
                total_contribution REAL NOT NULL DEFAULT 0.0,
                config_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type);
            CREATE INDEX IF NOT EXISTS idx_tasks_client ON tasks(client_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);"
        )?;

        Ok(Self { conn: Mutex::new(conn) })
    }

    /// 注册新任务
    pub fn register(&self, task: &Task) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO tasks (task_id, client_id, task_type, description, status, created_at, updated_at, rounds_participated, total_contribution, config_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                task.task_id, task.client_id, task.task_type, task.description,
                task.status, task.created_at, task.updated_at,
                task.rounds_participated, task.total_contribution, task.config_json
            ],
        )?;
        Ok(())
    }

    /// 获取任务
    pub fn get(&self, task_id: &str) -> Result<Option<Task>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT task_id, client_id, task_type, description, status, created_at, updated_at, rounds_participated, total_contribution, config_json FROM tasks WHERE task_id = ?1"
        )?;
        let task = stmt.query_row(params![task_id], |row| {
            Ok(Task {
                task_id: row.get(0)?,
                client_id: row.get(1)?,
                task_type: row.get(2)?,
                description: row.get(3)?,
                status: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
                rounds_participated: row.get(7)?,
                total_contribution: row.get(8)?,
                config_json: row.get(9)?,
            })
        }).ok();
        Ok(task)
    }

    /// 列出任务（支持过滤）
    pub fn list(&self, task_type: Option<&str>, status: Option<&str>) -> Result<Vec<Task>> {
        let conn = self.conn.lock().unwrap();
        let mut sql = String::from(
            "SELECT task_id, client_id, task_type, description, status, created_at, updated_at, rounds_participated, total_contribution, config_json FROM tasks WHERE 1=1"
        );
        if task_type.is_some() {
            sql.push_str(" AND task_type = ?");
        }
        if status.is_some() {
            sql.push_str(" AND status = ?");
        }
        sql.push_str(" ORDER BY updated_at DESC");

        let mut stmt = conn.prepare(&sql)?;
        let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(tt) = task_type {
            params_vec.push(Box::new(tt.to_string()));
        }
        if let Some(s) = status {
            params_vec.push(Box::new(s.to_string()));
        }
        let params_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

        let tasks = stmt.query_map(params_refs.as_slice(), |row| {
            Ok(Task {
                task_id: row.get(0)?,
                client_id: row.get(1)?,
                task_type: row.get(2)?,
                description: row.get(3)?,
                status: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
                rounds_participated: row.get(7)?,
                total_contribution: row.get(8)?,
                config_json: row.get(9)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;

        Ok(tasks)
    }

    /// 更新任务状态
    pub fn update_status(&self, task_id: &str, status: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE tasks SET status = ?1, updated_at = ?2 WHERE task_id = ?3",
            params![status, now, task_id],
        )?;
        Ok(())
    }

    /// 增加参与轮次
    pub fn increment_rounds(&self, task_id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE tasks SET rounds_participated = rounds_participated + 1, updated_at = ?1 WHERE task_id = ?2",
            params![now, task_id],
        )?;
        Ok(())
    }

    /// 增加贡献分数
    pub fn add_contribution(&self, task_id: &str, score: f64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE tasks SET total_contribution = total_contribution + ?1, updated_at = ?2 WHERE task_id = ?3",
            params![score, now, task_id],
        )?;
        Ok(())
    }

    /// 获取贡献排行榜
    pub fn leaderboard(&self, top_k: i32, task_type: Option<&str>) -> Result<Vec<(String, String, f64, i32)>> {
        let conn = self.conn.lock().unwrap();
        let mut sql = String::from(
            "SELECT client_id, task_type, total_contribution, rounds_participated FROM tasks"
        );
        if task_type.is_some() {
            sql.push_str(" WHERE task_type = ?");
        }
        sql.push_str(" ORDER BY total_contribution DESC LIMIT ?");

        let mut stmt = conn.prepare(&sql)?;
        let rows: Vec<(String, String, f64, i32)> = if let Some(tt) = task_type {
            stmt.query_map(params![tt, top_k], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?.collect::<Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![top_k], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?.collect::<Result<Vec<_>, _>>()?
        };
        Ok(rows)
    }

    /// 统计
    pub fn stats(&self) -> Result<(i32, i32)> {
        let conn = self.conn.lock().unwrap();
        let total: i32 = conn.query_row("SELECT COUNT(*) FROM tasks", [], |r| r.get(0))?;
        let active: i32 = conn.query_row("SELECT COUNT(*) FROM tasks WHERE status = 'active'", [], |r| r.get(0))?;
        Ok((total, active))
    }
}
