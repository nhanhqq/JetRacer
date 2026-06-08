import os
import glob
import shutil
import random
import zipfile
from ultralytics import YOLO

def train_model_v2():
    zip_path = '/home/nhanhq/JetRacer/traffic-sign.zip'
    base_img_dir = '/home/nhanhq/JetRacer/train'
    out_dir = '/home/nhanhq/JetRacer/yolo_dataset_v2'
    tmp_zip_dir = '/home/nhanhq/JetRacer/tmp_traffic_sign'

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
        '0_Go_straight': 'go-straight',
        '1_Turn_left': 'turn-left',
        '2_Turn_right': 'turn-right',
        '3_Prohibited': 'one-way'
    }

    dataset = []

    for img_folder, label_folder in classes.items():
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
                dataset.append((img_path, txt_path))

    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    def copy_data(data, split):
        for img_path, txt_path in data:
            img_name = os.path.basename(img_path)
            txt_name = os.path.basename(txt_path)
            shutil.copy(img_path, os.path.join(out_dir, f'images/{split}', img_name))
            shutil.copy(txt_path, os.path.join(out_dir, f'labels/{split}', txt_name))

    copy_data(train_data, 'train')
    copy_data(val_data, 'val')

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

    model = YOLO('/home/nhanhq/JetRacer/runs/detect/gtsrb_pretrain/weights/best.pt')
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        name='traffic_sign_detection_v2'
    )

if __name__ == '__main__':
    train_model_v2()
