"""
Embodied-FL 多客户端并行模拟

同时启动3个工厂客户端，模拟真实的分布式联邦训练场景。
每个客户端在独立进程中运行，通过 REST API 与服务器交互。
"""

import subprocess
import sys
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Embodied-FL Multi-Client Simulation")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--api", type=str, default="http://localhost:8080/api/v1")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially instead of parallel")
    args = parser.parse_args()

    factories = [
        ("factory-a", "苏州电子厂(SMT产线)", "inspection", "500", "42"),
        ("factory-b", "无锡汽车厂(抓取工位)", "grasping", "400", "123"),
        ("factory-c", "昆山3C厂(装配线)", "assembly", "350", "456"),
    ]

    print("=" * 60)
    print("  Embodied-FL Multi-Client Simulation")
    print("  Factories: 苏州电子厂 | 无锡汽车厂 | 昆山3C厂")
    print("=" * 60)

    if args.sequential:
        # 顺序执行（调试用）
        for cid, name, task, samples, seed in factories:
            print(f"\n🚀 Starting {name}...")
            subprocess.run([
                sys.executable, "client.py",
                "--rounds", str(args.rounds),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
                "--api", args.api,
            ], cwd=__import__("pathlib").Path(__file__).parent)
    else:
        # 并行执行
        processes = []
        for cid, name, task, samples, seed in factories:
            print(f"\n🚀 Starting {name}...")
            p = subprocess.Popen([
                sys.executable, "client.py",
                "--rounds", str(args.rounds),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
                "--api", args.api,
            ], cwd=__import__("pathlib").Path(__file__).parent)
            processes.append((name, p))

        # 等待所有进程完成
        for name, p in processes:
            p.wait()
            print(f"✅ {name} finished (exit code: {p.returncode})")

    print("\n" + "=" * 60)
    print("  All factories completed. Check dashboard →")
    print("  http://localhost:8080")
    print("=" * 60)


if __name__ == "__main__":
    main()
