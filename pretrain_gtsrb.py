from ultralytics import YOLO

def pretrain():
    model = YOLO('/home/nhanhq/JetRacer/yolo26n.pt')
    results = model.train(
        data='/home/nhanhq/JetRacer/gtsrb.yaml',
        epochs=10,
        imgsz=128,
        batch=64,
        device=0,
        name='gtsrb_pretrain'
    )

if __name__ == '__main__':
    pretrain()
