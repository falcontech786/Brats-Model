import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mod_233_custom_datagen",
    os.path.join(os.path.dirname(__file__), "233_custom_datagen.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

imageLoader = _mod.imageLoader
load_img = _mod.load_img
