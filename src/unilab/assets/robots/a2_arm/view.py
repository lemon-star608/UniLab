#!/usr/bin/env python
"""打开 MuJoCo viewer 查看 A2 + Airbot(含 sliderbase) 安装效果。

用法（用 UniLab 的 venv python 运行）:
    python view.py                 # motor 版，静态 home 姿态（默认，适合看安装位置）
    python view.py position        # position 伺服版，静态
    python view.py motor --sim     # 跑物理仿真（腿无 PD 会下沉，正常现象）

鼠标操作: 左键拖拽=旋转视角 / 右键拖拽=平移 / 滚轮=缩放 / 双击=聚焦。
关闭窗口即退出。
"""
import os
import sys
import time
import mujoco
import mujoco.viewer

here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
variant = "position" if "position" in args else "motor"
do_sim = "--sim" in args
scene = os.path.join(here, f"scene_{variant}.xml")

m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)

# 加载 home 关键帧（A2 站姿 + Airbot 折叠基线）
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
if kid >= 0:
    mujoco.mj_resetDataKeyframe(m, d, kid)
mujoco.mj_forward(m, d)

mode = "物理仿真" if do_sim else "静态 home 姿态（冻结物理，方便观察挂载）"
print(f"[view] 加载 {os.path.basename(scene)} | 模式: {mode}")
print("[view] 鼠标左键旋转 / 右键平移 / 滚轮缩放，关闭窗口退出")

with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        if do_sim:
            mujoco.mj_step(m, d)
            time.sleep(m.opt.timestep)
        else:
            # 只 forward 不 step：保持 home 姿态不动，便于静态查看安装效果
            mujoco.mj_forward(m, d)
            time.sleep(0.02)
        v.sync()
