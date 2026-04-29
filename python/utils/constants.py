# ── python/utils/constants.py ──
"""Shared constants for Embodied-FL Python modules."""

# Robot task types
TASK_TYPES = [
    "grasping", "navigation", "inspection",
    "assembly", "manipulation", "welding",
    "data_collection", "vla_training", "custom",
]

# Application domains
DOMAINS = [
    "manufacturing", "warehouse", "healthcare",
    "agriculture", "construction", "logistics", "custom",
]

# Sensor modalities
SENSORS = [
    "rgb_camera", "depth_camera", "lidar",
    "force_torque", "imu", "tactile", "custom",
]

# Factory presets
FACTORY_PRESETS = {
    "suzhou_electronics": {
        "name": "苏州电子厂 (SMT产线)",
        "task_type": "inspection",
        "domain": "manufacturing",
        "sensor": "rgb_camera",
        "classes": ["pcb", "component", "solder_joint", "defect", "conveyor"],
    },
    "wuxi_automotive": {
        "name": "无锡汽车厂 (抓取工位)",
        "task_type": "grasping",
        "domain": "manufacturing",
        "sensor": "depth_camera",
        "classes": ["car_part", "fixture", "tool", "human_worker", "safety_zone"],
    },
    "kunshan_3c": {
        "name": "昆山3C厂 (装配线)",
        "task_type": "assembly",
        "domain": "manufacturing",
        "sensor": "rgb_camera",
        "classes": ["phone_frame", "screw", "connector", "cable", "defect"],
    },
}

# Colors for visualization
COLORS = {
    "primary": "#38bdf8",
    "secondary": "#8b5cf6",
    "accent": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "bg_dark": "#0a0e1a",
    "bg_card": "#111827",
    "text": "#e2e8f0",
    "text_muted": "#64748b",
}

# Aggregation strategies
AGG_STRATEGIES = ["fedavg", "task_aware", "multi_task", "fedprox"]
