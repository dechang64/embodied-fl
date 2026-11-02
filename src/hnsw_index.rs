use anyhow::Result;
use hnsw::{Hnsw, Searcher, Params};
use rand_pcg::Pcg64;
use space::Metric;

/// Custom Euclidean distance metric for f32 vectors.
struct Euclidean;

impl Metric<Vec<f32>> for Euclidean {
    type Unit = u64;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u64 {
        let dist: f32 = a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        dist.to_bits() as u64
    }
}

/// HNSW vector index for task embedding similarity search.
pub struct HnswIndex {
    index: Hnsw<Euclidean, Vec<f32>, Pcg64, 12, 24>,
    dimension: usize,
    ids: Vec<String>,
}

impl HnswIndex {
    pub fn new(dimension: usize, _max_elements: usize, ef_construction: usize, _m: usize) -> Self {
        let params = Params::new().ef_construction(ef_construction);
        let index = Hnsw::new_params(Euclidean, params);
        Self {
            index,
            dimension,
            ids: Vec::new(),
        }
    }

    pub fn with_defaults(dimension: usize) -> Self {
        Self::new(dimension, 10000, 200, 16)
    }

    pub fn insert(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        assert_eq!(vector.len(), self.dimension, "Vector dimension mismatch");
        let mut searcher = Searcher::default();
        self.index.insert(vector.to_vec(), &mut searcher);
        self.ids.push(id.to_string());
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<(String, f32)>> {
        assert_eq!(query.len(), self.dimension, "Query dimension mismatch");
        let mut searcher = Searcher::default();
        let mut dest: Vec<space::Neighbor<u64>> = (0..k)
            .map(|_| space::Neighbor {
                index: 0,
                distance: 0,
            })
            .collect();
        let found = self.index.nearest(&query.to_vec(), ef_search, &mut searcher, &mut dest);
        let results: Vec<(String, f32)> = found
            .iter()
            .filter_map(|neighbor| {
                self.ids.get(neighbor.index).map(|id| {
                    let dist = f32::from_bits(neighbor.distance as u32);
                    (id.clone(), dist)
                })
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
