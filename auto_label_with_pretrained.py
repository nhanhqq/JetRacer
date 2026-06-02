import os
import glob
import shutil
from ultralytics import YOLO

def create_dataset_with_pretrained(pretrained_model_path, base_dir, out_dir):
    model = YOLO(pretrained_model_path)

    img_out_dir = os.path.join(out_dir, 'images/train')
    lbl_out_dir = os.path.join(out_dir, 'labels/train')

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    classes = {
        '0_Go_straight': 0,
        '1_Turn_left': 1,
        '2_Turn_right': 2,
        '3_Prohibited': 3
    }

    success_count = 0
    fail_count = 0

    for folder, custom_class_id in classes.items():
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue

        images = glob.glob(os.path.join(folder_path, '*.png'))
        for img_path in images:
            img_name = os.path.basename(img_path)
            unique_name = f"c{custom_class_id}_{img_name}"

            results = model.predict(img_path, verbose=False)
            boxes = results[0].boxes

            if len(boxes) > 0:
                best_box = max(boxes, key=lambda x: x.conf[0].item())
                b = best_box.xywhn[0].tolist()
                x_center, y_center, w_norm, h_norm = b

                lbl_path = os.path.join(lbl_out_dir, unique_name.replace('.png', '.txt'))
                with open(lbl_path, 'w') as f:
                    f.write(f"{custom_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                out_img_path = os.path.join(img_out_dir, unique_name)
                shutil.copy(img_path, out_img_path)

                success_count += 1
            else:
                fail_count += 1

    print(f"Dataset generated at {out_dir}")
    print(f"Labeled: {success_count} images")
    print(f"Failed: {fail_count} images")

if __name__ == '__main__':
    PRETRAINED_MODEL = '/home/nhanhq/JetRacer/runs/detect/gtsrb_pretrain/weights/best.pt'
    create_dataset_with_pretrained(
        pretrained_model_path=PRETRAINED_MODEL,
        base_dir='/home/nhanhq/JetRacer/train',
        out_dir='/home/nhanhq/JetRacer/yolo_dataset_pretrained'
    )
