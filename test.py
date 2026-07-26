import warnings
from functools import partial

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator


class FFTBandValidator(DetectionValidator):
    """Detection validator that applies FFT band filtering to images before inference."""

    def __init__(self, *args, pass_type: str = "high", cutoff: float = 0.25, **kwargs):
        self.pass_type = pass_type
        self.cutoff = float(cutoff)
        super().__init__(*args, **kwargs)

    def fft_band_filter(self, img: torch.Tensor) -> torch.Tensor:
        """Apply a high-pass or low-pass FFT filter to a batch of images.

        Args:
            img (torch.Tensor): Image tensor of shape [B, C, H, W] in range [0, 255].

        Returns:
            torch.Tensor: Filtered image tensor of shape [B, C, H, W].
        """
        img = img.to(dtype=torch.float32)
        freq = torch.fft.fftshift(torch.fft.fft2(img, dim=(-2, -1)), dim=(-2, -1))
        _, _, h, w = img.shape

        yy, xx = torch.meshgrid(
            torch.arange(h, device=img.device, dtype=img.dtype),
            torch.arange(w, device=img.device, dtype=img.dtype),
            indexing="ij",
        )
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        threshold = max(radius.max().item(), 1.0) * self.cutoff

        if self.pass_type == "high":
            mask = radius >= threshold
        else:
            mask = radius <= threshold

        mask = mask.to(dtype=img.dtype)[None, None, :, :]
        filtered = freq * mask
        restored = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), s=(h, w), dim=(-2, -1))
        return restored.real

    def preprocess(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")

        img = batch["img"].float()
        img = self.fft_band_filter(img)
        img = img / 255.0
        batch["img"] = img.half() if self.args.half else img
        return batch


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    device = 0
    # weights = 'runs/detect/spike-module-v1-12-12IFv4t2-l-bs16/weights/best.pt'
    weights = 'runs/detect/spike-module-v1-12-12IFv4-l-bs16/weights/best.pt'
    data = 'VOC.yaml'

    model = YOLO(weights)
    print(model.info())

    # metrics = model.val(
    #     # validator=partial(FFTBandValidator, pass_type=pass_type, cutoff=0.25),
    #     data=data,
    #     device=device)


    # for pass_type in ('high', 'low'):
    #     print(f"\nEvaluating baseline-l-bs16 on VOC with {pass_type}-frequency input")
    #     metrics = model.val(
    #         validator=partial(FFTBandValidator, pass_type=pass_type, cutoff=0.25),
    #         data=data,
    #         device=device,
    #     )
    #     # print(metrics)

    # model.val(data='VOC.yaml',device=device,imgsz=448,iou=0.5,conf=0.3)

    # model.val(data='DAWN_All.yaml',device=device,)
    # model.val(data='DAWN_Fog.yaml',device=device,imgsz=448,iou=0.5,conf=0.3)
    # model.val(data='DAWN_Rain.yaml',device=device,imgsz=448,iou=0.5,conf=0.3)
    # model.val(data='DAWN_Sand.yaml',device=device,imgsz=448,iou=0.5,conf=0.3)
    # model.val(data='DAWN_Snow.yaml',device=device,imgsz=448,iou=0.5,conf=0.3)

    # model.val(data='ExDark.yaml',device=device,imgsz=448,iou=0.5,conf=0.3)

    # model.val(data='RTTS.yaml',device=device,imgsz=448,iou=0.5,conf=0.3)

    model.val(data='VOC.yaml',device=device)
    # model.val(data='DAWN_All.yaml',device=device,)

    # model.val(data='ExDark.yaml',device=device)

    # model.val(data='RTTS.yaml',device=device)


    model.val(data='VOC_brightness1.yaml',device=device,)
    model.val(data='VOC_brightness3.yaml',device=device,)
    model.val(data='VOC_brightness5.yaml',device=device,)
    # model.val(data='VOC_contrast1.yaml',device=device,)
    # model.val(data='VOC_contrast3.yaml',device=device,)
    # model.val(data='VOC_contrast5.yaml',device=device,)
    model.val(data='VOC_fog1.yaml',device=device,)
    model.val(data='VOC_fog3.yaml',device=device,)
    model.val(data='VOC_fog5.yaml',device=device,)
    model.val(data='VOC_frost1.yaml',device=device,)
    model.val(data='VOC_frost3.yaml',device=device,)
    model.val(data='VOC_frost5.yaml',device=device,)
    model.val(data='VOC_snow1.yaml',device=device,)
    model.val(data='VOC_snow3.yaml',device=device,)
    model.val(data='VOC_snow5.yaml',device=device,)
