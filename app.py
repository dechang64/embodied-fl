"""Embodied-FL Streamlit Cloud entry point.

Adds python/ to sys.path so that analysis/, core/, utils/ imports resolve,
then delegates to streamlit_app.py.
"""
import sys
import os

# Ensure python/ subdirectory is on the import path
_python_dir = os.path.join(os.path.dirname(__file__), "python")
if os.path.isdir(_python_dir) and _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

# Delegate to the main Streamlit app
_app_file = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
with open(_app_file) as _f:
    _code = compile(_f.read(), _app_file, "exec")
exec(_code)
