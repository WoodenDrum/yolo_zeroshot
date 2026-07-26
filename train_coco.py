from ultralytics import YOLO

# from custom_trainer import CustomYOLO11Trainer

if __name__ == "__main__":
    # 加载预训练模型
    # model = YOLO("yolo11l-spike.yaml")


    # model = YOLO("yolo11l-spike.yaml")
    model = YOLO("yolo11l-spike-12.yaml")
    # model = YOLO("yolo11l.yaml")
    # model = YOLO("yolo11l.yaml")
    # model = YOLO("rtdetr-l.yaml",task="detect")
    # model = YOLO("rtdetr-l-spike-12.yaml",task="detect")
    # model = YOLO("runs/detect/baseline-l-coco-bs16/weights/last.pt")
    
    # 训练模型，指定数据集配置文件和训练轮数

    model.train(
        # model= "",
        data="coco.yaml",
        # data="VOC.yaml",
        epochs=300,
        imgsz=640,
        # batch=16,
        batch=16,
        workers=8,
        device=3,
        cache = 'disk',
        resume = True,
        # name="baseline-l-coco-bs16",
        name="STS-l-coco-bs16",
        # name="rtdetr-l-coco-bs4",
        # name="rtdetrspike-module-v1-12-12IFv4t2-l-coco-bs16",
    )
