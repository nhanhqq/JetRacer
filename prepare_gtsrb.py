import os
import glob
import cv2
import pandas as pd

def prepare_gtsrb_yolo():
    base_dir = '/tmp/gtsrb/gtsrb/GTSRB/Training'
    out_dir = '/home/nhanhq/JetRacer/gtsrb_yolo_dataset'

    img_out = os.path.join(out_dir, 'images/train')
    lbl_out = os.path.join(out_dir, 'labels/train')

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    csv_files = glob.glob(os.path.join(base_dir, '*/*.csv'))
    count = 0

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=';')
        folder = os.path.dirname(csv_file)

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

            x_center = (x1 + x2) / 2.0 / w_img
            y_center = (y1 + y2) / 2.0 / h_img
            w_norm = (x2 - x1) / float(w_img)
            h_norm = (y2 - y1) / float(h_img)

            folder_name = os.path.basename(folder)
            unique_name_base = f"{folder_name}_{img_filename.replace('.ppm', '')}"

            lbl_path = os.path.join(lbl_out, f"{unique_name_base}.txt")
            with open(lbl_path, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            out_img_path = os.path.join(img_out, f"{unique_name_base}.jpg")
            img = cv2.imread(img_path)
            
            if img is not None:
                cv2.imwrite(out_img_path, img)
                count += 1

    print(f"Prepared {count} images and labels in {out_dir}")

if __name__ == '__main__':
    prepare_gtsrb_yolo()
