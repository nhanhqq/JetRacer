from ultralytics import YOLO

def train_model():
    model = YOLO('/home/nhanhq/JetRacer/runs/detect/gtsrb_pretrain/weights/best.pt')
    results = model.train(
        data='/home/nhanhq/JetRacer/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        name='traffic_sign_detection'
    )
    
if __name__ == '__main__':
    train_model()
