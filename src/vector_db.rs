use anyhow::Result;
use crate::hnsw_index::HnswIndex;
use std::collections::HashMap;

/// 向量数据库 — 管理任务特征向量和模型嵌入
pub struct VectorDb {
    index: HnswIndex,
    vectors: HashMap<String, Vec<f32>>,
    metadata: HashMap<String, HashMap<String, String>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub meta: HashMap<String, String>,
}

impl VectorDb {
    pub fn new(dimension: usize) -> Self {
        Self {
            index: HnswIndex::with_defaults(dimension),
            vectors: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: &str, vector: &[f32], meta: Option<HashMap<String, String>>) -> Result<()> {
        if self.vectors.contains_key(id) {
            self.vectors.insert(id.to_string(), vector.to_vec());
            if let Some(m) = meta { self.metadata.insert(id.to_string(), m); }
            self.rebuild_index()?;
        } else {
            self.index.insert(id, vector)?;
            self.vectors.insert(id.to_string(), vector.to_vec());
            if let Some(m) = meta { self.metadata.insert(id.to_string(), m); }
        }
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let ef_search = std::cmp::max(k * 4, 50);
        let raw = self.index.search(query, k, ef_search)?;
        Ok(raw.into_iter().map(|(id, distance)| {
            let meta = self.metadata.get(&id).cloned().unwrap_or_default();
            SearchResult { id, distance, meta }
        }).collect())
    }

    pub fn get(&self, id: &str) -> Option<&Vec<f32>> { self.vectors.get(id) }
    pub fn len(&self) -> usize { self.vectors.len() }
    pub fn is_empty(&self) -> bool { self.vectors.is_empty() }

    fn rebuild_index(&mut self) -> Result<()> {
        let dimension = self.index.dimension();
        let mut new_index = HnswIndex::with_defaults(dimension);
        for (id, vector) in &self.vectors { new_index.insert(id, vector)?; }
        self.index = new_index;
        Ok(())
    }
}
