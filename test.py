import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    device = 0
    # pth = '/home/yzx/wab/dww/ultralytics/runs/detect/C3k2Universal5+convwithgray3-shortcutTT-bs16/weights/best.pt'
    # pth = '/disk/duanww/ultralytics/runs/detect/C3k2Universal14in1234-TTTT-bs16/weights/best.pt'
    pth = "/disk/duanww/ultralytics/runs/detect/Bottleneck_SpikeAttention_V9-bs32/weights/best.pt"
    # pth = '/home/yzx/wab/dww/ultralytics/runs/detect/baseline-bs16/weights/best.pt'

    model = YOLO(pth)
    print(model.info())
    print(model.info(detailed=True))
    model.val(
        data="VOC.yaml",
        device=device,
    )

    model.val(
        data="DAWN_All.yaml",
        device=device,
    )
    # model.val(data='DAWN_Fog.yaml',device=device,)
    # model.val(data='DAWN_Rain.yaml',device=device,)
    # model.val(data='DAWN_Sand.yaml',device=device,)
    # model.val(data='DAWN_Snow.yaml',device=device,)

    model.val(
        data="ExDark.yaml",
        device=device,
    )

    model.val(
        data="RTTS.yaml",
        device=device,
    )

    # model.val(data='VOC_brightness1.yaml',device=device,)
    # model.val(data='VOC_brightness3.yaml',device=device,)
    # model.val(data='VOC_brightness5.yaml',device=device,)
    # model.val(data='VOC_contrast1.yaml',device=device,)
    # model.val(data='VOC_contrast3.yaml',device=device,)
    # model.val(data='VOC_contrast5.yaml',device=device,)
    # model.val(data='VOC_fog1.yaml',device=device,)
    # model.val(data='VOC_fog3.yaml',device=device,)
    # model.val(data='VOC_fog5.yaml',device=device,)
    # model.val(data='VOC_frost1.yaml',device=device,)
    # model.val(data='VOC_frost3.yaml',device=device,)
    # model.val(data='VOC_frost5.yaml',device=device,)
    # model.val(data='VOC_snow1.yaml',device=device,)
    # model.val(data='VOC_snow3.yaml',device=device,)
    # model.val(data='VOC_snow5.yaml',device=device,)
