"""pym3t の sparse-viewpoint テンプレート（*.region.meta / *.depth.meta）を事前生成する。
unit_in_meter が変わると pym3t は起動時に再生成する（1 モデル 1〜2 分）ので、
デモ直前に待たされないよう先にまとめて作っておく。
使い方:  <pym3t env python> regen_templates.py [obj ...]   （引数なし = arch + 4 groups）
"""
import os
import sys
import time

_conda_lib_bin = os.path.join(sys.prefix, "Library", "bin")
if os.name == "nt" and os.path.isdir(_conda_lib_bin):
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_conda_lib_bin)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYM3T_DIR = os.path.abspath(os.environ.get("PYM3T_DIR", os.path.join(os.path.dirname(SCRIPT_DIR), "pym3t")))
sys.path.insert(0, PYM3T_DIR)
import numpy as np
import pym3t

MODEL_UNIT_IN_METER = 0.025
names = sys.argv[1:] or ["arch_4groups_smooth.obj", "group_1_molar_smooth.obj", "group_2_premolar_canine_smooth.obj",
                         "group_3_canine_lateral_smooth.obj", "group_4_incisor_smooth.obj"]
K = np.array([[1001.6, 0, 640.2], [0, 1024.0, 334.5], [0, 0, 1]], dtype=np.float32)
for name in names:
    path = os.path.join(SCRIPT_DIR, "models", name)
    t0 = time.time()
    tracker = pym3t.Tracker(1280, 720, K, use_region=True, use_depth=False, use_texture=False)
    model = pym3t.Model(name, path, "", "", MODEL_UNIT_IN_METER, 0.8)
    tracker.add_model(model)   # setup → メタが古ければここで再生成される
    print(f"{name}: ready in {time.time() - t0:.0f} s", flush=True)
print("done")
