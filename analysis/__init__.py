# ── python/analysis/__init__.py ──
# Lazy imports to avoid requiring heavy deps (ultralytics, transformers, torchvision)
# at module load time. Each module is imported only when accessed.

def __getattr__(name):
    _lazy = {
        'RobotSceneDetector': '.detector',
        'DINOv2SceneExtractor': '.feature_extractor',
        'MetadataFallbackExtractor': '.feature_extractor',
        'get_extractor': '.feature_extractor',
        'GradCAM': '.gradcam',
        'generate_robot_explanation': '.gradcam',
        'EmbodiedMultiTaskFL': '.multi_task_fl',
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
