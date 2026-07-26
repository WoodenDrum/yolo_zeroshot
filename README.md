# 3Spike

Open-source research implementation accompanying the **3Spike** paper.

This repository is a modified derivative of [Ultralytics](https://github.com/ultralytics/ultralytics). It retains the upstream source required to run the experiments, together with the 3Spike module and model configurations. It is an independent research repository and is not affiliated with, endorsed by, or sponsored by Ultralytics.

## Paper code

The paper-facing `STS` module is defined in `ultralytics/nn/modules/block.py`, exported by `ultralytics/nn/modules/__init__.py`, and registered for YAML model parsing in `ultralytics/nn/tasks.py`.

The supplied paper configurations are:

- `ultralytics/cfg/models/11/yolo11-spike-12.yaml`
- `ultralytics/cfg/models/12/yolo12-spike-12.yaml`
- `ultralytics/cfg/models/26/yolo26-spike-12.yaml`
- `ultralytics/cfg/models/rt-detr/rtdetr-l-spike-12.yaml`
- `ultralytics/cfg/models/v3/yolov3-spike-12.yaml`

Other experimental variants are retained for reproducibility but are not necessarily part of the paper results.

## Installation

Use a clean Python environment, install a PyTorch build appropriate for your platform, then install this project in editable mode:

```bash
pip install -e .
```

Refer to the selected model configuration and the experiment scripts in the repository when reproducing results. Training data, pretrained weights, generated outputs, and local caches are intentionally not included.

## License and attribution

3Spike is distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**, as required for this derivative work. The complete license text is in [LICENSE](LICENSE). You must preserve the license and the notices in [NOTICE](NOTICE) when redistributing or modifying this code.

Ultralytics and the Ultralytics YOLO source code remain the property of their respective owners. Please cite both the 3Spike paper (when available) and the upstream Ultralytics project in academic work that uses this repository.
