from ultralytics import YOLO

# from custom_trainer import CustomYOLO11Trainer

if __name__ == "__main__":
    # 加载预训练模型
    # model = YOLO("yolo11l-spike.yaml")


    # model = YOLO("yolo11l-spike.yaml")
    # model = YOLO("yolo11l-spike-12.yaml")
    # model = YOLO("yolo11l.yaml")
    # model = YOLO("yolov8l.yaml")
    # model = YOLO("yolo12l.yaml")
    # model = YOLO("yolov8l-spike-12.yaml")
    # model = YOLO("yolo26l-spike-12.yaml")
    # model = YOLO("rtdetr-l-spike-12.yaml",task="detect")
    model = YOLO("runs/detect/rtdetr-l-coco-bs4/weights/last.pt")
    
    # 训练模型，指定数据集配置文件和训练轮数

    model.train(
        # model= "",
        data="coco.yaml",
        # seed=1,
        # data="VOC.yaml",
        epochs=200,
        imgsz=640,
        batch=4,
        # batch=4,
        workers=4,
        device=1,
        # cache = 'disk',
        # name="baseline-l-withoutNewLoss-bs16",
        # name="baseline8-l-bs16",
        # name="baseline26-l-bs16",
        # name="baseline12-l-bs16",
        # name="rtdetr-l-bs4",
        # name="rtdetrspike-module-v1-12-12IFv4t2-l-bs16",
        # name="26spike-module-v1-12-12IFv4t2-l-bs16",
        # name="spike-module-v1-12-12IF-step12-l-bs16",
        # name="spike-module-v1-12-12IFv4t2gatet2-l-bs16",
        resume=True,
        # name="spike-module-v1-12-12IF-l-voc2012-bs16"
        # name = "MorphFilter+STSG-bs32",
        # name="C3k2Universal14in123+ReliabilityGateFusion2-bs16",
        # name="C3k2Universal14in123-TTT-bs32",
        # trainer=CustomYOLO11Trainer
    )
