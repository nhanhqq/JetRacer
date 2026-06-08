import os
import random
from ultralytics import YOLO

def inference():
    model = YOLO('/home/nhanhq/JetRacer/runs/detect/traffic_sign_detection_v2/weights/best.pt')
    val_dir = '/home/nhanhq/JetRacer/yolo_dataset_v2/images/val'
    
    images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.png')]
    img_path = random.choice(images)
    
    results = model(img_path)
    results[0].save('/home/nhanhq/JetRacer/output.png')

if __name__ == '__main__':
    inference()
