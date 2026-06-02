#!/bin/bash

set -e

echo "1. Preparing GTSRB dataset..."
python3 prepare_gtsrb.py

echo "2. Pre-training model on GTSRB..."
python3 pretrain_gtsrb.py

echo "3. Auto-labeling custom JetRacer dataset..."
python3 auto_label_with_pretrained.py

echo "4. Training final model..."
python3 train_model.py

echo "All done. Final model: runs/detect/traffic_sign_detection/weights/best.pt"
echo "To test: python3 detect_sign.py --image <path_to_image>"
