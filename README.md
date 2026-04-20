<div align="center">

# Embodied-FL

### Federated Learning Platform for Embodied Intelligence

**Distributed robot training with data privacy — data never leaves the factory.**

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

[Paper](docs/paper.md) · [Architecture](docs/architecture.md) · [Simulation Guide](docs/simulation.md)

</div>

---

## 🎯 Problem

Embodied AI (humanoid robots, industrial arms, autonomous vehicles) requires massive amounts of training data. But:

| Barrier | Example |
|---------|---------|
| **Data silos** | Each factory's production data is proprietary (NDA-protected) |
| **Privacy** | Home robots capture private environments (cameras, layouts, habits) |
| **Regulation** | Medical/care robots handle patient data (HIPAA, GDPR) |
| **Domain shift** | Different production lines → different defect distributions |
| **Data ownership** | Who owns the data? Who profits from it? |

**Result**: Each company trains in isolation on limited data → suboptimal models.

## 💡 Solution

**Embodied-FL** enables multiple factories/robots to collaboratively train AI models **without sharing raw data**:

```
Factory A (SMT)     Factory B (Auto)     Factory C (3C)
  🏭 PCB检测          🏭 零件抓取          🏭 精密装配
     │                   │                   │
     ▼                   ▼                   ▼
  Local Training      Local Training      Local Training
     │                   │                   │
     │  gradients only   │  gradients only   │  gradients only
     ▼                   ▼                   ▼
  ┌─────────────────────────────────────────────────┐
  │           FedServer (Rust + gRPC)               │
  │  Task-aware aggregation + HNSW matching         │
  │  + Blockchain audit + Contribution tracking      │
  └─────────────────────────────────────────────────┘
     │
     ▼
  Better global model → deployed back to all factories
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🦀 **Rust Core** | High-performance gRPC server, HNSW vector search, SQLite storage |
| 🤖 **Heterogeneous Tasks** | Different robots doing different tasks (grasping, navigation, assembly) can collaborate |
| 🎯 **Task-Aware Aggregation** | HNSW vector search matches similar tasks → smarter weighted averaging than FedAvg |
| 🔐 **Blockchain Audit** | SHA-256 hash chain records every operation — immutable, verifiable |
| 📊 **Contribution Tracking** | Quantifies each factory's data contribution → basis for data pricing |
| 🌐 **Web Dashboard** | Real-time monitoring of training rounds, client status, leaderboard |
| 🐍 **Python Simulation** | 3-factory simulation with pure NumPy (no PyTorch required) |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Web Dashboard (Axum)                     │
│                  http://localhost:8080                        │
├──────────────────────────────────────────────────────────────┤
│                    REST API (Axum)                            │
│          /api/v1/stats  /tasks  /leaderboard  /audit         │
├──────────────────────────────────────────────────────────────┤
│                    gRPC Service (Tonic)                       │
│          FederatedService · TaskRegistry · Contribution       │
├────────────┬──────────────┬──────────────┬───────────────────┤
│ TaskRegistry│  FedServer   │ Contribution │    VectorDb      │
│ (SQLite)    │  (SQLite)    │  Tracker     │   (HNSW)         │
│             │              │  (SQLite)    │                   │
│ task types  │ round mgmt   │ score calc   │ task embedding    │
│ task match  │ model versn  │ leaderboard  │ similarity search │
├────────────┴──────────────┴──────────────┴───────────────────┤
│                    AuditChain (SHA-256)                       │
│              Immutable operation log                          │
└──────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │ gRPC / REST        │ gRPC / REST        │ gRPC / REST
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    │Factory A│          │Factory B│          │Factory C│
    │(Python) │          │(Python) │          │(Python) │
    │SMT检测  │          │零件抓取  │          │精密装配  │
    └─────────┘          └─────────┘          └─────────┘
```

## 🚀 Quick Start

### 1. Build & Run Server

```bash
git clone https://github.com/dechang64/embodied-fl.git
cd embodied-fl
cargo run
# gRPC server ready on 0.0.0.0:50051
# REST server ready on 0.0.0.0:8080
# Web dashboard: http://0.0.0.0:8080
```

### 2. Run Simulation

```bash
# Terminal 2: Install dependencies
cd python/sim
pip install -r ../../python/requirements.txt

# Run 3-factory federated training simulation
python run_all.py --rounds 10 --epochs 5

# Or run a single factory
python client.py --rounds 10 --epochs 5 --lr 0.01
```

### 3. Monitor

Open **http://localhost:8080** to see:
- Training progress (loss/accuracy per round)
- Active clients and their status
- Contribution leaderboard
- Audit chain verification

## 🧪 Simulation Scenario

| Factory | Location | Task | Data | Description |
|---------|----------|------|------|-------------|
| A | 苏州电子厂 | PCB Inspection | 500 samples | SMT 产线缺陷检测 |
| B | 无锡汽车厂 | Part Grasping | 400 samples | 机械臂零件抓取 |
| C | 昆山3C厂 | Precision Assembly | 350 samples | 精密装配任务 |

Each factory trains a **simple MLP policy network** (pure NumPy, no GPU needed):
- Input: 24-dim state vector (joint angles + object pose + force/torque)
- Hidden: 64 neurons, ReLU
- Output: 6-dim action vector (joint velocities)

**Federated training loop:**
1. Each factory downloads the global model
2. Trains locally for N epochs on its own data
3. Uploads gradients to server
4. Server aggregates (FedAvg with task-aware weighting)
5. Repeat

## 📊 Expected Results

After 10 rounds of federated training:

```
Round   Avg Loss    Avg Acc
    1     0.2341     0.4123
    2     0.1987     0.4756
    3     0.1654     0.5389
    ...
   10     0.0523     0.8234
```

Each factory benefits from the collective knowledge of the other two, even though they do different tasks.

## 🔬 Research Contributions

1. **Task-Aware Federated Aggregation**: Unlike standard FedAvg (uniform weighting), Embodied-FL uses HNSW to find task similarity and weights accordingly
2. **Contribution Quantification**: Blockchain-audited contribution scores enable fair data pricing
3. **Heterogeneous Task Federation**: Different robot tasks can collaborate through shared representation learning

## 🤝 Related Projects

| Project | Domain | Link |
|---------|--------|------|
| [organoid-fl](https://github.com/dechang64/organoid-fl) | Medical organoid image analysis | Federated learning + Rust HNSW |
| [defect-fl](https://github.com/dechang64/defect-fl) | PCB defect detection | Federated learning + Rust HNSW |
| [FundFL](https://github.com/dechang64/fundfl) | Private fund analysis | Vector search + risk metrics |

Embodied-FL shares ~60% of its core infrastructure (HNSW, gRPC, audit chain) with these projects.

## 📄 License

Apache-2.0

---

<div align="center">

**Embodied-FL** — Train robots together, keep data apart.

</div>
