import os
import glob
import shutil
import random
import zipfile
import cv2
import pandas as pd
from ultralytics import YOLO

def prepare_gtsrb():
    base_dir = '/tmp/gtsrb/gtsrb/GTSRB/Training'
    out_dir = '/home/nhanhq/JetRacer/gtsrb_yolo_v2'
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
    img_out = os.path.join(out_dir, 'images/train')
    lbl_out = os.path.join(out_dir, 'labels/train')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    
    class_map = {
        35: 0,
        34: 1,
        33: 2,
        17: 3
    }
    
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
                
            w_img = row['Width']
            h_img = row['Height']
            x1 = row['Roi.X1']
            y1 = row['Roi.Y1']
            x2 = row['Roi.X2']
            y2 = row['Roi.Y2']
            
            cx = (x1 + x2) / 2.0 / w_img
            cy = (y1 + y2) / 2.0 / h_img
            w_norm = (x2 - x1) / float(w_img)
            h_norm = (y2 - y1) / float(h_img)
            
            unique_name = f"gtsrb_{class_id}_{img_filename.replace('.ppm', '')}"
            img = cv2.imread(img_path)
            
            if img is not None:
                cv2.imwrite(os.path.join(img_out, f"{unique_name}.jpg"), img)
                with open(os.path.join(lbl_out, f"{unique_name}.txt"), 'w') as f:
                    f.write(f"{yolo_class} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    yaml_path = os.path.join(out_dir, 'gtsrb.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")
        f.write("nc: 4\n")
        f.write("names:\n")
        f.write("  0: Go_straight\n")
        f.write("  1: Turn_left\n")
        f.write("  2: Turn_right\n")
        f.write("  3: Prohibited\n")
        
    return yaml_path

def prepare_jetracer():
    zip_path = '/home/nhanhq/JetRacer/traffic-sign.zip'
    base_img_dir = '/home/nhanhq/JetRacer/train'
    out_dir = '/home/nhanhq/JetRacer/jetracer_yolo_v2'
    tmp_zip_dir = '/home/nhanhq/JetRacer/tmp_traffic_sign_v2'

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    if os.path.exists(tmp_zip_dir):
        shutil.rmtree(tmp_zip_dir)

    os.makedirs(tmp_zip_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_zip_dir)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(out_dir, f'images/{split}'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, f'labels/{split}'), exist_ok=True)

    classes = {
        '0_Go_straight': ('go-straight', 0),
        '1_Turn_left': ('turn-left', 1),
        '2_Turn_right': ('turn-right', 2),
        '3_Prohibited': ('one-way', 3)
    }

    dataset = []

    for img_folder, (label_folder, true_class_id) in classes.items():
        img_folder_path = os.path.join(base_img_dir, img_folder)
        lbl_folder_path = os.path.join(tmp_zip_dir, 'traffic-sign', label_folder)

        if not os.path.exists(img_folder_path):
            continue

        images = glob.glob(os.path.join(img_folder_path, '*.png'))
        for img_path in images:
            img_name = os.path.basename(img_path)
            txt_name = img_name.replace('.png', '.txt')
            txt_path = os.path.join(lbl_folder_path, txt_name)

            if os.path.exists(txt_path):
                dataset.append((img_path, txt_path, true_class_id))

    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    def process_data(data, split):
        for img_path, txt_path, true_class_id in data:
            img_name = os.path.basename(img_path)
            txt_name = os.path.basename(txt_path)
            shutil.copy(img_path, os.path.join(out_dir, f'images/{split}', img_name))
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            with open(os.path.join(out_dir, f'labels/{split}', txt_name), 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        f.write(f"{true_class_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")

    process_data(train_data, 'train')
    process_data(val_data, 'val')

    yaml_path = os.path.join(out_dir, 'data.yaml')
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

    shutil.rmtree(tmp_zip_dir)
    return yaml_path

def pipeline():
    gtsrb_yaml = prepare_gtsrb()
    
    model = YOLO('/home/nhanhq/JetRacer/yolo26n.pt')
    model.train(
        data=gtsrb_yaml,
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        name='gtsrb_pretrain_v2'
    )
    
    jetracer_yaml = prepare_jetracer()
    
    model = YOLO('/home/nhanhq/JetRacer/runs/detect/gtsrb_pretrain_v2/weights/best.pt')
    model.train(
        data=jetracer_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        name='traffic_sign_finetune_v2'
    )

if __name__ == '__main__':
    pipeline()
