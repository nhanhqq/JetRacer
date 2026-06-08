import cv2
from ultralytics import YOLO

def run_livecam():
    model = YOLO('/home/nhanhq/JetRacer/runs/detect/traffic_sign_finetune_v2/weights/best.pt')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
            
        results = model(frame)
        annotated_frame = results[0].plot()
        
        cv2.imshow("Live Inference", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_livecam()
