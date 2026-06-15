# ── analysis/__init__.py ──
from .gradcam import GradCAM, generate_robot_explanation
from .multi_task_fl import EmbodiedMultiTaskFL

# Heavy-dependency modules — lazy import for Streamlit Cloud compatibility
try:
    from .detector import RobotSceneDetector
except (ImportError, ModuleNotFoundError):
    RobotSceneDetector = None

try:
    from .feature_extractor import DINOv2SceneExtractor, MetadataFallbackExtractor, get_extractor
except (ImportError, ModuleNotFoundError):
    DINOv2SceneExtractor = None
    MetadataFallbackExtractor = None
    get_extractor = None
