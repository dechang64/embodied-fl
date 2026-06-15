"""
Embodied-FL — 具身智能联邦检测平台
Streamlit Cloud App

Features:
    - Upload robot scene image
    - AI object detection (factory-specific classes)
    - DINOv2 scene feature extraction (mock)
    - Multi-task federated learning simulation
    - Multi-factory dashboard
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time
import os

from analysis.detector import RobotSceneDetector, Detection
from analysis.multi_task_fl import EmbodiedMultiTaskFL
from utils.constants import (
    TASK_TYPES, DOMAINS, SENSORS, FACTORY_PRESETS,
    COLORS, AGG_STRATEGIES,
)


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Embodied-FL · 具身智能联邦检测",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #666;
    }
    .severity-critical { background: #ef4444; color: white; }
    .severity-major { background: #f59e0b; color: white; }
    .severity-minor { background: #22c55e; color: white; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
if "detector" not in st.session_state:
    st.session_state.detector = RobotSceneDetector(mode="mock")
if "last_detections" not in st.session_state:
    st.session_state.last_detections = []
if "last_image" not in st.session_state:
    st.session_state.last_image = None


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/robot.png", width=64)
    st.title("具身智能")
    st.caption("Embodied-FL v0.1")

    st.divider()

    st.subheader("⚙️ 设置")
    detect_mode = st.selectbox(
        "检测模式",
        ["mock (快速演示)", "yolo (需要GPU)"],
        index=0,
    )
    st.session_state.detector = RobotSceneDetector(
        mode="mock" if "mock" in detect_mode else "yolo"
    )

    st.divider()

    st.subheader("🏭 工厂配置")
    factory_key = st.selectbox(
        "工厂预设",
        list(FACTORY_PRESETS.keys()),
        format_func=lambda k: FACTORY_PRESETS[k]["name"],
        index=0,
    )
    factory = FACTORY_PRESETS[factory_key]

    st.caption(f"**任务类型**: {factory['task_type']}")
    st.caption(f"**应用领域**: {factory['domain']}")
    st.caption(f"**传感器**: {factory['sensor']}")
    st.caption(f"**检测类别**: {', '.join(factory['classes'])}")

    st.divider()

    st.subheader("📋 任务信息")
    task_type = st.selectbox("任务类型", TASK_TYPES, index=TASK_TYPES.index(factory["task_type"]))
    domain = st.selectbox("应用领域", DOMAINS, index=DOMAINS.index(factory["domain"]))
    sensor = st.selectbox("传感器", SENSORS, index=SENSORS.index(factory["sensor"]))


# ============================================================
# Main Page
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🤖 Embodied-FL</h1>
    <p>基于联邦学习的多工厂具身智能场景检测平台</p>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_fl, tab_about = st.tabs([
    "🔍 场景检测", "🤝 联邦学习", "ℹ️ 关于"
])


# ============================================================
# Tab 1: Scene Detection
# ============================================================
with tab_detect:
    st.header("机器人场景智能检测")
    st.caption(f"当前工厂: {factory['name']} | 检测类别: {', '.join(factory['classes'])}")

    uploaded = st.file_uploader(
        "上传场景图像",
        type=["png", "jpg", "jpeg", "webp"],
        key="detect_upload",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("原始场景")
            st.image(img, use_container_width=True)
            st.caption(f"尺寸: {img.width} × {img.height} px")

        with col2:
            with st.spinner("🔍 正在检测场景物体..."):
                t0 = time.time()
                detections = st.session_state.detector.detect(img_array, factory_key)
                dt = (time.time() - t0) * 1000

            # Draw detections
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)

            # Generate colors for classes
            class_colors = {}
            palette = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
                       "#06b6d4", "#ec4899", "#f97316", "#14b8a6", "#6366f1"]
            for i, cls in enumerate(factory["classes"]):
                class_colors[cls] = palette[i % len(palette)]

            for d in detections:
                color = class_colors.get(d.class_name, "#888")
                draw.rectangle([d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]], outline=color, width=2)
                label = f"{d.class_name} {d.confidence:.0%}"
                draw.rectangle([d.bbox[0], d.bbox[1]-16, d.bbox[0]+len(label)*9+6, d.bbox[1]], fill=color)
                draw.text((d.bbox[0]+3, d.bbox[1]-14), label, fill="white")

            st.subheader("检测结果")
            st.image(annotated, use_container_width=True)
            st.caption(f"检测耗时: {dt:.0f}ms")

        # Store
        st.session_state.last_detections = detections
        st.session_state.last_image = img_array

        # Summary
        summary = st.session_state.detector.summary(detections)
        st.divider()
        st.subheader("📊 检测统计")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("检测总数", summary["total"])
        with m2:
            st.metric("平均置信度", f"{summary['avg_confidence']:.1%}")
        with m3:
            st.metric("平均面积", f"{summary['avg_area']:.0f} px²")

        # Class breakdown
        if summary["classes"]:
            st.subheader("分类分布")
            cols = st.columns(min(len(summary["classes"]), 5))
            for i, (cls_name, count) in enumerate(summary["classes"].items()):
                with cols[i % len(cols)]:
                    color = class_colors.get(cls_name, "#888")
                    st.markdown(f"""
                    <div style="background:{color}; color:white; padding:0.5rem; border-radius:8px; text-align:center;">
                        <div style="font-size:1.5rem; font-weight:bold;">{count}</div>
                        <div style="font-size:0.85rem;">{cls_name}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Detection table
        if detections:
            st.subheader("详细结果")
            table_data = []
            for i, d in enumerate(detections):
                table_data.append({
                    "#": i + 1,
                    "类型": d.class_name,
                    "置信度": f"{d.confidence:.1%}",
                    "中心": f"({d.cx:.0f}, {d.cy:.0f})",
                    "宽×高": f"{d.width:.0f}×{d.height:.0f}",
                    "面积": f"{d.area:.0f} px²",
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("请上传机器人场景图像开始检测")


# ============================================================
# Tab 2: Federated Learning
# ============================================================
with tab_fl:
    st.header("多任务联邦学习模拟")
    st.caption("多工厂协同训练：检测 + 分类 + 策略三任务并行")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ 训练参数")
        n_rounds = st.number_input("联邦轮数", value=10, min_value=1, max_value=50)
        n_clients = st.number_input("工厂数量", value=3, min_value=2, max_value=10)
        lr = st.number_input("学习率", value=0.001, min_value=0.0001, max_value=0.1, format="%.4f")
        local_epochs = st.number_input("本地轮数", value=2, min_value=1, max_value=10)
        agg_strategy = st.selectbox("聚合策略", AGG_STRATEGIES, index=0)
        n_samples = st.number_input("每厂样本数", value=500, min_value=100, max_value=5000, step=100)

        run_fl = st.button("🚀 开始训练", type="primary", use_container_width=True)

    with col2:
        if run_fl:
            with st.spinner("🤝 多任务联邦训练中..."):
                engine = EmbodiedMultiTaskFL(
                    input_dim=768,
                    num_classes=len(factory["classes"]),
                    hidden_dim=128,
                    lr=lr,
                    local_epochs=local_epochs,
                )

                progress = st.empty()
                def on_progress(rnd, metrics):
                    progress.text(f"轮次 {rnd}/{n_rounds} — 验证准确率: {metrics['val_acc']:.1%}")

                # Generate synthetic data
                rng = np.random.RandomState(42)
                n_cls = len(factory["classes"])
                features = rng.randn(n_samples * n_clients, 768).astype(np.float32)
                labels = rng.randint(0, n_cls, size=n_samples * n_clients)

                history = engine.run(
                    features=features,
                    labels=labels,
                    n_clients=n_clients,
                    rounds=n_rounds,
                    progress_callback=on_progress,
                )

            st.success("✅ 训练完成！")

            final = history[-1]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("最终训练准确率", f"{final['avg_train_acc']:.1%}")
            with m2:
                st.metric("最终验证准确率", f"{final['val_acc']:.1%}")
            with m3:
                st.metric("最终验证损失", f"{final['val_loss']:.3f}")
            with m4:
                st.metric("聚合策略", agg_strategy)

            # Convergence chart
            st.subheader("📈 收敛曲线")
            rounds = [h["round"] for h in history]
            chart_data = {
                "轮次": rounds,
                "训练准确率": [h["avg_train_acc"] for h in history],
                "验证准确率": [h["val_acc"] for h in history],
                "训练损失": [h["avg_train_loss"] for h in history],
                "验证损失": [h["val_loss"] for h in history],
            }
            st.line_chart(chart_data, x="轮次", y=["训练准确率", "验证准确率"])

            # Per-client
            st.subheader("🏭 各工厂训练详情")
            for cm in final["client_metrics"]:
                fname = list(FACTORY_PRESETS.keys())[cm["client_id"] % len(FACTORY_PRESETS)]
                fi = FACTORY_PRESETS[fname]
                st.markdown(f"**{fi['name']}** — 样本: {cm['n_samples']:,} | 损失: {cm['train_loss']:.3f} | 准确率: {cm['train_acc']:.1%}")
        else:
            st.info("配置参数后点击「开始训练」")


# ============================================================
# Tab 3: About
# ============================================================
with tab_about:
    st.header("关于 Embodied-FL")

    st.markdown("""
    ### 🤖 项目简介

    **Embodied-FL** 是一个基于联邦学习的多工厂具身智能场景检测平台，实现跨工厂协同训练，生产数据不出厂。

    ### 🔬 核心技术

    | 技术 | 用途 |
    |------|------|
    | YOLOv11 | 多类别物体检测 |
    | DINOv2 | 768维场景特征提取 |
    | Multi-Task FL | 检测+分类+策略三任务并行 |
    | Task-Aware | 基于任务相似度的加权聚合 |
    | HNSW | 任务匹配与检索 |

    ### 🏭 工厂预设

    | 工厂 | 任务 | 传感器 | 检测类别 |
    |------|------|--------|----------|
    | 苏州电子厂 | 质检 | RGB相机 | PCB/元件/焊点/缺陷/传送带 |
    | 无锡汽车厂 | 抓取 | 深度相机 | 零件/夹具/工具/工人/安全区 |
    | 昆山3C厂 | 装配 | RGB相机 | 手机框/螺丝/连接器/线缆/缺陷 |

    ### 🎯 任务类型

    - **抓取** (Grasping) — 机械臂抓取
    - **导航** (Navigation) — 自主导航
    - **质检** (Inspection) — 质量检测
    - **装配** (Assembly) — 自动装配
    - **操作** (Manipulation) — 精细操作
    - **焊接** (Welding) — 自动焊接

    ### 📄 开源协议

    Apache-2.0 | [GitHub](https://github.com/dechang64)
    """)
