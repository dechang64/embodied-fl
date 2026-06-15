use anyhow::Result;
use std::path::Path;

/// HNSW 向量索引（基于 hnsw crate）
///
/// 用于任务嵌入的快速近似最近邻搜索，
/// 实现任务感知的联邦聚合加权。
pub struct HnswIndex {
    index: hnsw::Hnsw<f32, space::Euclidean>,
    dimension: usize,
    max_elements: usize,
    ids: Vec<String>,
}

impl HnswIndex {
    pub fn new(dimension: usize, max_elements: usize, ef_construction: usize, m: usize) -> Self {
        let config = hnsw::Config {
            max_elements,
            ef_construction,
            m,
            ..Default::default()
        };
        let index = hnsw::Hnsw::new(config);
        Self {
            index,
            dimension,
            max_elements,
            ids: Vec::with_capacity(max_elements),
        }
    }

    pub fn with_defaults(dimension: usize) -> Self {
        Self::new(dimension, 10000, 200, 16)
    }

    pub fn insert(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        assert_eq!(vector.len(), self.dimension, "Vector dimension mismatch");
        if self.ids.len() >= self.max_elements {
            anyhow::bail!("HNSW index full: {} / {} elements", self.ids.len(), self.max_elements);
        }
        self.ids.push(id.to_string());
        self.index.insert(vector.to_vec());
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<(String, f32)>> {
        assert_eq!(query.len(), self.dimension, "Query dimension mismatch");
        let neighbors = self.index.search(query, ef_search, k);
        let results: Vec<(String, f32)> = neighbors
            .into_iter()
            .filter_map(|(idx, dist)| {
                self.ids.get(idx).map(|id| (id.clone(), dist))
            })
            .collect();
        Ok(results)
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}
