# ── python/analysis/__init__.py ──
from .detector import RobotSceneDetector
from .feature_extractor import DINOv2SceneExtractor, MetadataFallbackExtractor, get_extractor
from .gradcam import GradCAM, generate_robot_explanation
from .multi_task_fl import EmbodiedMultiTaskFL
