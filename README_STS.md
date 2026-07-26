# 3Spike paper-code map

This file maps the paper implementation onto the upstream-derived codebase.

- `STS` is the paper-facing alias for `NightVisionModulatorIFv4t2`.
- The implementation is in `ultralytics/nn/modules/block.py`.
- It is exported in `ultralytics/nn/modules/__init__.py` and registered in `ultralytics/nn/tasks.py` so it can be referenced from YAML model files.

Paper configurations:

- `ultralytics/cfg/models/26/yolo26-spike-12.yaml`
- `ultralytics/cfg/models/11/yolo11-spike-12.yaml`
- `ultralytics/cfg/models/12/yolo12-spike-12.yaml`
- `ultralytics/cfg/models/rt-detr/rtdetr-l-spike-12.yaml`
- `ultralytics/cfg/models/v3/yolov3-spike-12.yaml`

This repository is a modified derivative of Ultralytics and is licensed as AGPL-3.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
