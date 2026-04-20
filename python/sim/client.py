"""
Embodied-FL 模拟客户端 — 模拟工厂机械臂参与联邦训练

场景：3个工厂，各有不同任务
  - 工厂A：电子厂 SMT 产线，PCB 缺陷检测（inspection）
  - 工厂B：汽车厂，零件抓取（grasping）
  - 工厂C：3C 装配厂，精密装配（assembly）

每个工厂本地训练一个简单的 MLP 策略网络，
通过 REST API 与联邦服务器交互：下载全局模型 → 本地训练 → 上传梯度。
"""

import numpy as np
import requests
import json
import time
import argparse
from pathlib import Path

API_BASE = "http://localhost:8080/api/v1"


class SimplePolicyNet:
    """简单的 MLP 策略网络（纯 NumPy，无 PyTorch 依赖）"""

    def __init__(self, input_dim=24, hidden_dim=64, output_dim=6):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Xavier 初始化
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        """前向传播"""
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def compute_loss(self, x, y):
        """MSE Loss"""
        pred = self.forward(x)
        return np.mean((pred - y) ** 2)

    def compute_gradients(self, x, y):
        """反向传播，返回梯度"""
        m = x.shape[0]
        pred = self.forward(x)

        # 输出层梯度
        dz2 = 2 * (pred - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # 隐藏层梯度
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)  # ReLU 导数
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return [dW1, db1, dW2, db2]

    def apply_gradients(self, grads, lr=0.01):
        """应用梯度"""
        for param, grad in zip(self.params, grads):
            param -= lr * grad

    def get_weights(self):
        """序列化权重"""
        return [w.tolist() for w in self.params]

    def set_weights(self, weights):
        """反序列化权重"""
        for param, w in zip(self.params, weights):
            param[:] = np.array(w)

    def accuracy(self, x, y):
        """计算准确率（离散化预测）"""
        pred = self.forward(x)
        pred_labels = np.argmax(pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        return np.mean(pred_labels == true_labels)


class FactorySimulator:
    """工厂模拟器 — 生成合成训练数据"""

    def __init__(self, factory_id, task_type, num_samples=500, seed=42):
        self.factory_id = factory_id
        self.task_type = task_type
        self.rng = np.random.RandomState(seed)
        self.num_samples = num_samples

        # 根据任务类型生成不同的数据分布
        if task_type == "inspection":
            # PCB 缺陷检测：输入=图像特征(24d)，输出=缺陷分类(6类)
            self.input_dim = 24
            self.output_dim = 6
            self.X = self.rng.randn(num_samples, self.input_dim) * 0.5
            # 添加任务特定的模式
            self.X[:, :6] += self.rng.choice([-1, 0, 1], size=(num_samples, 6))
            self.y = self._one_hot(self.rng.randint(0, 6, num_samples), 6)
            self.description = "PCB defect inspection on SMT production line"

        elif task_type == "grasping":
            # 零件抓取：输入=传感器数据(24d)，输出=抓取策略(6类)
            self.input_dim = 24
            self.output_dim = 6
            self.X = self.rng.randn(num_samples, self.input_dim) * 0.8
            self.X[:, 12:18] += self.rng.randn(num_samples, 6) * 0.3  # 力觉传感器
            self.y = self._one_hot(self.rng.randint(0, 6, num_samples), 6)
            self.description = "Robotic arm grasping for automotive parts"

        elif task_type == "assembly":
            # 精密装配：输入=视觉+力觉(24d)，输出=装配动作(6类)
            self.input_dim = 24
            self.output_dim = 6
            self.X = self.rng.randn(num_samples, self.input_dim) * 0.6
            self.X[:, 6:12] += self.rng.randn(num_samples, 6) * 0.4  # 视觉特征
            self.y = self._one_hot(self.rng.randint(0, 6, num_samples), 6)
            self.description = "Precision assembly for 3C electronics"

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _one_hot(self, labels, num_classes):
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot

    def sample_batch(self, batch_size=32):
        """随机采样一个 batch"""
        idx = self.rng.choice(self.num_samples, batch_size, replace=False)
        return self.X[idx], self.y[idx]


class FederatedClient:
    """联邦学习客户端 — 与服务器交互"""

    def __init__(self, client_id, client_name, task_type, num_samples=500, seed=42):
        self.client_id = client_id
        self.client_name = client_name
        self.task_type = task_type
        self.simulator = FactorySimulator(client_id, task_type, num_samples, seed)
        self.model = SimplePolicyNet(
            input_dim=self.simulator.input_dim,
            output_dim=self.simulator.output_dim
        )
        self.prev_loss = float('inf')

    def register(self):
        """注册客户端"""
        resp = requests.post(f"{API_BASE}/clients/register", json={
            "client_id": self.client_id,
            "client_name": self.client_name,
            "task_type": self.task_type,
        })
        print(f"  [{self.client_id}] Registered: {resp.json()}")

    def register_task(self):
        """注册任务"""
        resp = requests.post(f"{API_BASE}/tasks", json={
            "client_id": self.client_id,
            "task_type": self.task_type,
            "description": self.simulator.description,
            "config": {
                "input_dim": self.simulator.input_dim,
                "output_dim": self.simulator.output_dim,
                "num_samples": self.simulator.num_samples,
            }
        })
        print(f"  [{self.client_id}] Task registered: {resp.json()}")

    def train_local(self, epochs=5, lr=0.01, batch_size=32):
        """本地训练"""
        losses = []
        for epoch in range(epochs):
            x_batch, y_batch = self.simulator.sample_batch(batch_size)
            grads = self.model.compute_gradients(x_batch, y_batch)
            self.model.apply_gradients(grads, lr)
            loss = self.model.compute_loss(self.simulator.X, self.simulator.y)
            losses.append(loss)
        return losses

    def report_update(self, round_id):
        """上报训练结果"""
        loss = self.model.compute_loss(self.simulator.X, self.simulator.y)
        acc = self.model.accuracy(self.simulator.X, self.simulator.y)
        improvement = self.prev_loss - loss

        resp = requests.post(f"{API_BASE}/rounds/{round_id}/update", json={
            "client_id": self.client_id,
            "task_type": self.task_type,
            "num_samples": self.simulator.num_samples,
            "local_loss": loss,
            "local_accuracy": acc,
            "weights": self.model.get_weights(),
        })

        self.prev_loss = loss
        return {"loss": loss, "accuracy": acc, "improvement": improvement, "response": resp.json()}

    def run_fed_round(self, round_id, local_epochs=5, lr=0.01):
        """执行一轮联邦训练"""
        print(f"  [{self.client_id}] Training locally ({local_epochs} epochs, lr={lr})...")
        losses = self.train_local(local_epochs, lr)
        result = self.report_update(round_id)
        print(f"  [{self.client_id}] Loss: {result['loss']:.4f}, Acc: {result['accuracy']:.4f}, Δ: {result['improvement']:.4f}")
        return result


def run_simulation(num_rounds=10, local_epochs=5, lr=0.01):
    """运行完整的联邦学习模拟"""

    print("=" * 60)
    print("  Embodied-FL Simulation")
    print("  3 Factories × Federated Training × Data Never Leaves")
    print("=" * 60)

    # 创建3个工厂客户端
    factories = [
        FederatedClient("factory-a", "苏州电子厂 (SMT产线)", "inspection", num_samples=500, seed=42),
        FederatedClient("factory-b", "无锡汽车厂 (抓取工位)", "grasping", num_samples=400, seed=123),
        FederatedClient("factory-c", "昆山3C厂 (装配线)", "assembly", num_samples=350, seed=456),
    ]

    # 注册
    print("\n📡 Registering clients...")
    for f in factories:
        f.register()
        f.register_task()

    # 联邦训练循环
    print(f"\n🔄 Starting federated training ({num_rounds} rounds)...\n")
    history = []

    for round_num in range(1, num_rounds + 1):
        print(f"--- Round {round_num}/{num_rounds} ---")

        # 触发新一轮
        try:
            resp = requests.post(f"{API_BASE}/rounds/start", json={"round_num": round_num})
            round_data = resp.json()
            round_id = round_data.get("round_id", round_num)
        except Exception:
            round_id = round_num

        # 每个工厂本地训练并上报
        round_results = []
        for f in factories:
            result = f.run_fed_round(round_id, local_epochs, lr)
            round_results.append({
                "client_id": f.client_id,
                "task_type": f.task_type,
                **result
            })

        # 触发聚合
        try:
            agg_resp = requests.post(f"{API_BASE}/rounds/{round_id}/aggregate")
            agg_data = agg_resp.json()
            print(f"  📊 Aggregated — Loss: {agg_data.get('loss', 'N/A')}, Acc: {agg_data.get('accuracy', 'N/A')}")
        except Exception as e:
            print(f"  ⚠️  Aggregation skipped: {e}")

        avg_loss = np.mean([r["loss"] for r in round_results])
        avg_acc = np.mean([r["accuracy"] for r in round_results])
        history.append({"round": round_num, "avg_loss": avg_loss, "avg_acc": avg_acc, "results": round_results})
        print(f"  📈 Round {round_num} avg: loss={avg_loss:.4f}, acc={avg_acc:.4f}\n")

    # 打印最终结果
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\n{'Round':>6} {'Avg Loss':>10} {'Avg Acc':>10}")
    print("-" * 30)
    for h in history:
        print(f"{h['round']:>6} {h['avg_loss']:>10.4f} {h['avg_acc']:>10.4f}")

    print("\n🏆 Final per-factory results:")
    for f in factories:
        loss = f.model.compute_loss(f.simulator.X, f.simulator.y)
        acc = f.model.accuracy(f.simulator.X, f.simulator.y)
        print(f"  {f.client_name}: loss={loss:.4f}, acc={acc:.4f}")

    print("\n✅ Simulation complete. Check http://localhost:8080 for dashboard.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embodied-FL Simulation Client")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Local training epochs per round")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--api", type=str, default=API_BASE, help="Server API base URL")
    args = parser.parse_args()

    global API_BASE
    API_BASE = args.api

    run_simulation(num_rounds=args.rounds, local_epochs=args.epochs, lr=args.lr)
