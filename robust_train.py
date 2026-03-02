import copy

import torch
from ultralytics.models.yolo.detect import DetectionTrainer

from ultralytics.utils import DEFAULT_CFG


class RobustTrainer(DetectionTrainer):
    """自定义训练器，加入基于有限差分的梯度稀疏正则化 (SR)."""

    # 关键修正2：将 cfg 的默认值设为 DEFAULT_CFG，而不是 None
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 Trainer。 必须使用 DEFAULT_CFG 作为 cfg 的默认值，否则 get_cfg 会报错。.
        """
        if overrides is None:
            overrides = {}

        # 调用父类初始化
        super().__init__(cfg, overrides, _callbacks)

        # --- SR 算法超参数 ---
        self.sr_epsilon = 0.05  # 扰动步长
        self.sr_lambda = 0.5  # 正则化权重
        print(f"\n✅ RobustTrainer Ready | Epsilon: {self.sr_epsilon}, Lambda: {self.sr_lambda}\n")

    def train_step(self, batch):
        """重写单步训练逻辑，加入 SR 正则化."""
        self.optimizer.zero_grad()

        # 1. Clean Pass (原始图像前向传播)
        loss, loss_items = self.model(batch)

        # 2. SR Pass (梯度稀疏正则化)
        if self.sr_lambda > 0:
            # 复制 batch 以防修改原数据
            batch_noisy = copy.copy(batch)
            images = batch_noisy["img"]

            # 生成随机噪声 (-epsilon ~ +epsilon)
            noise = (torch.rand_like(images) * 2 - 1) * self.sr_epsilon
            batch_noisy["img"] = images + noise

            # Noisy Pass (加噪图像前向传播)
            loss_noisy, _ = self.model(batch_noisy)

            # 计算有限差分: |Loss_noisy - Loss_clean| / epsilon
            sr_term = torch.abs(loss_noisy - loss) / self.sr_epsilon

            # 叠加 Loss
            total_loss = loss + (self.sr_lambda * sr_term)

            # 反向传播
            total_loss.backward()

            # 更新日志显示的 Loss
            if isinstance(loss_items, torch.Tensor):
                loss_items[0] = total_loss.detach()

        else:
            loss.backward()

        self.optimizer.step()
        return loss, loss_items


# --- 启动训练 ---
if __name__ == "__main__":
    # -------------------------------------------------------------
    # 1. 准备参数
    # -------------------------------------------------------------
    training_args = {
        "model": "yolo11m-spike.yaml",
        "data": "VOC.yaml",
        "epochs": 200,
        "imgsz": 640,
        "batch": 32,
        "workers": 4,
        "device": 1,
    }

    try:
        trainer = RobustTrainer(overrides=training_args)
        trainer.train()

    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback

        traceback.print_exc()
