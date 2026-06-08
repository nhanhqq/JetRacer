import os
import glob
import shutil
import random
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

def build_gtsrb_yolo_dataset():
    base_dir = '/tmp/gtsrb/gtsrb/GTSRB/Training'
    out_dir = '/home/nhanhq/JetRacer/gtsrb_yolo_v3'
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
    for split in ['train', 'val']:
        os.makedirs(os.path.join(out_dir, f'images/{split}'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, f'labels/{split}'), exist_ok=True)
        
    class_map = {
        35: 0,
        34: 1,
        33: 2,
        17: 3
    }
    
    dataset = []
    csv_files = glob.glob(os.path.join(base_dir, '*/*.csv'))
    
    for csv_file in csv_files:
        folder = os.path.dirname(csv_file)
        class_id = int(os.path.basename(folder))
        
        if class_id not in class_map:
            continue
            
        yolo_class = class_map[class_id]
        df = pd.read_csv(csv_file, sep=';')
        
        for index, row in df.iterrows():
            img_filename = row['Filename']
            img_path = os.path.join(folder, img_filename)
            
            if not os.path.exists(img_path):
                continue
                
            x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
            dataset.append((img_path, yolo_class, x1, y1, x2, y2))
            
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    def process_and_save(data, split):
        bg_size = 640
        for img_path, yolo_class, x1, y1, x2, y2 in data:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            sign_img = img[y1:y2, x1:x2]
            if sign_img.size == 0:
                continue
                
            sh, sw = sign_img.shape[:2]
            
            scale = random.uniform(0.5, 4.0)
            new_w, new_h = int(sw * scale), int(sh * scale)
            
            if new_w > bg_size * 0.8 or new_h > bg_size * 0.8:
                scale_limit = (bg_size * 0.8) / max(sw, sh)
                new_w, new_h = int(sw * scale_limit), int(sh * scale_limit)
                
            sign_img = cv2.resize(sign_img, (new_w, new_h))
            sh, sw = sign_img.shape[:2]
            
            bg_color = (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))
            bg = np.zeros((bg_size, bg_size, 3), dtype=np.uint8)
            bg[:] = bg_color
            
            max_x = bg_size - sw
            max_y = bg_size - sh
            
            start_x = random.randint(0, max_x) if max_x > 0 else 0
            start_y = random.randint(0, max_y) if max_y > 0 else 0
            
            bg[start_y:start_y+sh, start_x:start_x+sw] = sign_img
            
            cx = (start_x + sw / 2.0) / bg_size
            cy = (start_y + sh / 2.0) / bg_size
            w_norm = sw / bg_size
            h_norm = sh / bg_size
            
            unique_name = f"gtsrb_{yolo_class}_{os.path.basename(img_path).replace('.ppm', '')}_{random.randint(0,100000)}"
            
            cv2.imwrite(os.path.join(out_dir, f'images/{split}', f"{unique_name}.jpg"), bg)
            with open(os.path.join(out_dir, f'labels/{split}', f"{unique_name}.txt"), 'w') as f:
                f.write(f"{yolo_class} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    process_and_save(train_data, 'train')
    process_and_save(val_data, 'val')
    
    yaml_path = os.path.join(out_dir, 'data_v3.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 4\n")
        f.write("names:\n")
        f.write("  0: Go_straight\n")
        f.write("  1: Turn_left\n")
        f.write("  2: Turn_right\n")
        f.write("  3: Prohibited\n")
        
    return yaml_path

def train_v3():
    yaml_path = build_gtsrb_yolo_dataset()
    model = YOLO('/home/nhanhq/JetRacer/yolo26n.pt')
    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        name='traffic_sign_v3'
    )

if __name__ == '__main__':
    train_v3()
