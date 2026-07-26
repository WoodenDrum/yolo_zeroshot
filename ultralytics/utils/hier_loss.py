import torch
import torch.nn as nn
import torch.nn.functional as F

def attention_map(x):
    """
    x: [B, C, H, W]
    return: [B, 1, H, W]
    """
    a = x.abs().mean(dim=1, keepdim=True)
    a = a / (a.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return a

def attention_consistency_loss(f_teacher, f_student):
    """
    teacher -> student
    """
    with torch.no_grad():
        a_t = attention_map(f_teacher)

    a_s = attention_map(f_student)

    if a_t.shape[-2:] != a_s.shape[-2:]:
        a_t = F.interpolate(a_t, size=a_s.shape[-2:], mode='bilinear', align_corners=False)

    return F.l1_loss(a_s, a_t)

class HierarchicalConsistencyLoss(nn.Module):
    """
    Configurable hierarchical attention consistency loss.
    """

    def __init__(self, config=None):
        super().__init__()
        default_config = {
            "enabled": True,
            "loss_type": "attention_l1",
            "pairs": [
                {"teacher": 14, "student": 12, "weight": 0.05, "enabled": True},
                {"teacher": 12, "student": 9,  "weight": 0.05, "enabled": True},
                {"teacher": 9,  "student": 6,  "weight": 0.10, "enabled": True},
                {"teacher": 6,  "student": 3,  "weight": 0.05, "enabled": True},
            ],
        }
        self.config = default_config if config is None else config

    def forward(self, feats):
        """
        feats: dict
        expected keys may include: 3, 6, 9, 12, 14
        """
        device = next(iter(feats.values())).device if len(feats) > 0 else "cpu"
        loss = torch.tensor(0.0, device=device)
        logs = {}

        if not self.config.get("enabled", False):
            logs["hier_loss_total"] = loss.detach()
            return loss, logs

        loss_type = self.config.get("loss_type", "attention_l1")
        if loss_type != "attention_l1":
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        pairs = self.config.get("pairs", [])

        for pair in pairs:
            t = pair["teacher"]
            s = pair["student"]
            w = pair.get("weight", 1.0)
            enabled = pair.get("enabled", True)

            if not enabled:
                continue
            if w <= 0:
                continue
            if t not in feats or s not in feats:
                continue

            l = attention_consistency_loss(feats[t], feats[s])
            loss = loss + w * l
            logs[f"{t}->{s}"] = l.detach()

        logs["hier_loss_total"] = loss.detach()
        return loss, logs