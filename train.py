from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型
    model = YOLO("yolo11m-spike.yaml")
    # model = YOLO("yolo11m.yaml")
    # model = YOLO("/disk/duanww/ultralytics/runs/detect/train-BMDStemv5/weights/last.pt")

    # 训练模型，指定数据集配置文件和训练轮数

    model.train(
        model="",
        data="VOC.yaml",
        epochs=200,
        imgsz=640,
        batch=32,
        workers=4,
        device=1,
        # name="baselineAllIN-bs32",
        name="Bottleneck_SpikeAttention_V9-bs32",
        # name="C3k2Universal14in123+ReliabilityGateFusion2-bs16",
        # name="C3k2Universal14in123-TTT-bs32",
    )
