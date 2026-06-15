<div align="center">

# Embodied-FL v2

### Federated Learning Platform for Embodied Intelligence

**Distributed robot training with data privacy — data never leaves the factory.**

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-9b59b6)](https://docs.ultralytics.com/)
[![DINOv2](https://img.shields.io/badge/DINOv2-Meta-blueviolet)](https://github.com/facebookresearch/dinov2)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

[Paper](paper/embodied-fl-preprint.pdf) · [Experiments](experiments/)

</div>

---

## 🎯 Problem

Embodied AI requires massive training data. But factory data is proprietary, home robots capture private environments, and regulations (HIPAA, GDPR) restrict data sharing.

**Result**: Each company trains in isolation → suboptimal models.

**Embodied-FL**: Train robots together, keep data apart.

---

## 🆕 What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| Policy Network | NumPy MLP | **PyTorch** (autograd, GPU) |
| Task Embedding | 32-dim one-hot | **768-dim DINOv2** (self-supervised) |
| Detection | YOLOv8 (separate) | **YOLOv11** (integrated) |
| Aggregation | FedAvg + Task-Aware | **Multi-Task FL** (detection + classification + policy) |
| Explainability | ❌ None | **Grad-CAM** (why did the robot decide this?) |
| Embedding Mode | Metadata only | **Metadata / Vision / Hybrid** |
| Strategy | FedAvg, Task-Aware | **+ Multi-Task, FedProx** |

### Architecture

```
Factory A ──┐                    ┌── YOLOv11 (scene detection)
            │    FedAvg          ├── DINOv2 + Classifier (task understanding)
Factory B ──┼──────────────────→ ├── Policy MLP (robot control)
            │    Aggregation     └── Grad-CAM (explainability)
Factory C ──┘
```

### Multi-Task FL

| Model | Task | Shared Weights | Frozen |
|-------|------|---------------|--------|
| YOLOv11 | Scene detection | Backbone only | Detection head |
| DINOv2 + Linear | Task classification | Linear head | Full ViT |
| Policy MLP | Robot control | Full MLP | — |

---

## 🏗️ Architecture

### Rust Backend (unchanged, bug-fixed)
- **FedServer**: Multi-task aggregation (FedAvg / Task-Aware / Multi-Task / FedProx)
- **TaskRegistry**: SQLite-backed task management
- **TaskEmbedding**: DINOv2 vision + metadata hybrid embedding
- **HNSW Index**: Fast task similarity search
- **ContributionTracker**: Data contribution quantification
- **AuditChain**: SHA-256 blockchain audit trail
- **gRPC + REST API**: Full service interface
- **Web Dashboard**: Real-time monitoring

### Python Client (upgraded)
- **PyTorch Policy Network**: Replaces NumPy MLP
- **DINOv2 Scene Extractor**: 768-dim self-supervised features
- **YOLOv11 Detector**: Robot scene object detection
- **Grad-CAM**: Decision explainability
- **Multi-Task FL Engine**: Detection + Classification + Policy

---

## 🚀 Quick Start

### Rust Server
```bash
cargo run                    # Start server (gRPC:50051, REST:8080)
```

### Python Client
```bash
pip install -r python/requirements.txt
python python/sim/client.py --rounds 10 --epochs 5
```

### Multi-Client Simulation
```bash
python python/sim/run_all.py --rounds 10 --parallel
```

---

## 📊 Experiments

6 experiments demonstrating embodied FL advantages:

| Exp | Scenario | FedAvg | Ours | Improvement |
|-----|----------|--------|------|-------------|
| 1 | 5 clients, Non-IID | 84.38% | 85.29% | +2.1% |
| 2 | 10 clients scalability | 82.1% | 83.5% | +1.7% |
| 3 | Non-IID severity sweep | — | — | Consistent |
| 4 | Heterogeneous tasks | 80.30% | **84.97%** | **+9.4%** |
| 5 | Continual learning (EWC) | 71.2% | 78.8% | +10.7% |
| 6 | Gradient compression (Top-K) | — | — | 10× with <2% loss |

---

## 🤝 Related Projects

| Project | Domain | Shared Infrastructure |
|---------|--------|---------------------|
| [organoid-fl](https://github.com/dechang64/organoid-fl) | Medical organoid analysis | HNSW, gRPC, audit, YOLO, DINOv2, SAM2 |
| [defect-fl](https://github.com/dechang64/defect-fl) | PCB defect detection | HNSW, gRPC, audit |
| [FundFL](https://github.com/dechang64/fundfl) | Private fund analysis | Vector search, risk metrics |

Embodied-FL v2 shares ~70% of its Python analysis code with organoid-fl.

---

## 📄 License

Apache-2.0

---

<div align="center">

**Embodied-FL v2** — Train robots together, keep data apart.

</div>
