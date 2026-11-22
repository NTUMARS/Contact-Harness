#!/usr/bin/env bash
# Render runs/<RUN>/episode.mp4 from runs/<RUN>/trajectory.npz
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
"$HERE/.venv_sim/bin/python" "$HERE/render_video.py" "$@" 2>&1 | grep -v "EGL\|Traceback\|File \|raise\|err =\|baseOperation\|cArguments\|<OpenGL\|result =\|^\s*)\|Exception ignored\|self\.\|return self"
