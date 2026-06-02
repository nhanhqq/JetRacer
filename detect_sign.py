import argparse
import cv2
import os
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Detect traffic signs in an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--weights", type=str, default="runs/detect/traffic_sign_detection/weights/best.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save the output image")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        return

    model = YOLO(args.weights)
    results = model.predict(source=args.image, save=False)
    res_img = results[0].plot()
    
    cv2.imwrite(args.output, res_img)
    print(f"Detection saved to {args.output}")

if __name__ == "__main__":
    main()
