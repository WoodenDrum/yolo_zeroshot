import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    device = 1

    pth = 'runs/detect/rtdetrspike-module-v1-12-12IFv4t2-l-coco-bs16/weights/best.pt'
    # pth = 'runs/detect/baseline-l-coco-bs32/weights/last.pt'

    model = YOLO(pth)
    print( model.info())
    # print( model.info(detailed=True) )
    model.val(data='coco.yaml',device=device,)
    # # # model.val(data='VOC2012.yaml',device=device,)

    model.val(data='DAWN_All_coco.yaml',device=device,)
    # model.val(data='DAWN_Fog.yaml',device=device,)
    # model.val(data='DAWN_Rain.yaml',device=device,)
    # model.val(data='DAWN_Sand.yaml',device=device,)
    # model.val(data='DAWN_Snow.yaml',device=device,)

    model.val(data='ExDark_coco.yaml',device=device,)

    model.val(data='RTTS_coco.yaml',device=device,)

    model.val(data='coco_brightness1.yaml',device=device,)
    model.val(data='coco_brightness3.yaml',device=device,)
    model.val(data='coco_brightness5.yaml',device=device,)
    model.val(data='coco_contrast1.yaml',device=device,)
    model.val(data='coco_contrast3.yaml',device=device,)
    model.val(data='coco_contrast5.yaml',device=device,)
    # model.val(data='coco_fog1.yaml',device=device,)
    # model.val(data='coco_fog3.yaml',device=device,)
    # model.val(data='coco_fog5.yaml',device=device,)
    # model.val(data='coco_frost1.yaml',device=device,)
    # model.val(data='coco_frost3.yaml',device=device,)
    # model.val(data='coco_frost5.yaml',device=device,)
    # model.val(data='coco_snow1.yaml',device=device,)
    # model.val(data='coco_snow3.yaml',device=device,)
    # model.val(data='coco_snow5.yaml',device=device,)
