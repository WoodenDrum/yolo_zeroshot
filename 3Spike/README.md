# 3Spike module

This directory isolates the research contribution of the 3Spike paper from the upstream-derived Ultralytics codebase.

- `module.py` contains the standalone `STS` implementation and its temporal-shift helper.
- `configs/` contains the five paper model configurations.

The runtime registration used by the Ultralytics YAML parser remains in `ultralytics/nn/modules/` for compatibility. `module.py` mirrors that registered implementation; keep the two copies synchronized if the module is changed.

## Dependencies

The module requires PyTorch and SpikingJelly:

```bash
pip install torch spikingjelly
```

This research code is distributed under AGPL-3.0. See the repository-level [LICENSE](../LICENSE) and [NOTICE](../NOTICE) for the upstream attribution and licensing terms.
