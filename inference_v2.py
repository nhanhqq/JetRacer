import os
import glob
import random
from ultralytics import YOLO

def inference():
    weight_files = glob.glob('/home/nhanhq/JetRacer/runs/detect/*/weights/best.pt')
    if not weight_files:
        return
    best_weight = max(weight_files, key=os.path.getmtime)
    model = YOLO(best_weight)
    val_dir = '/home/nhanhq/JetRacer/yolo_dataset_v2/images/val'
    
    images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.png')]
    img_path = random.choice(images)
    
    results = model(img_path)
    results[0].save('/home/nhanhq/JetRacer/output.png')

if __name__ == '__main__':
    inference()
