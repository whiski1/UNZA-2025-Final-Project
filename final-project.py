# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO

# # Load YOLOv8 model (use "yolov8s.pt" for better accuracy)
# model = YOLO("yolo11m-cls.pt")  # Use "yolov8n.pt" for ultra-fast speed

# # Open webcam (0 is the default camera)
# cap = cv2.VideoCapture(0)

# # Set resolution (optional, for speed tuning)
# cap.set(3, 640)  # Width
# cap.set(4, 480)  # Height

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # frame = cv2.GaussianBlur(frame, (3, 3), 0)
#     # frame = cv2.filter2D(frame, -1, sharpen_kernel)
#     # Perform detection with optimized parameters
#     results = model(frame, 
#                     conf=0.15,       # Confidence threshold (higher = fewer false positives)
#                     # iou=0.01,        # IoU threshold for NMS
#                     classes=[67],   # Only detect phones (class 67 in COCO)
#                     imgsz=416,      # Input image size (640x640 is best balance)
#                     max_det=2,      # Maximum detections per frame
#                     device="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
#                     # half=True       # Use FP16 for speed (if GPU supports it)
#                     )

#     # Plot detections on the frame
#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = box.conf[0].item()
#             cls = int(box.cls[0].item())

#             # Draw bounding box if it's a phone
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow("Phone Detector", frame)

#     # Exit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()




import cv2
from ultralytics import YOLO

seg_model = YOLO("yolo11n-seg.pt")  

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    seg_results = seg_model(frame, conf=0.2, classes=[67], verbose=False) 

    for r in seg_results:
        frame = r.plot()

    cv2.imshow("Phone Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



# from mmdet.apis import init_detector, inference_detector
# import cv2

# # Load model
# config_file = "configs/yolo/yolov5_l.py"
# checkpoint_file = "checkpoints/yolov5_l.pth"
# model = init_detector(config_file, checkpoint_file, device="cuda:0")

# # Open webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect phones
#     results = inference_detector(model, frame)

#     # Draw detections (Phones are class 67 in COCO)
#     for bbox in results[67]:  
#         x1, y1, x2, y2, score = bbox
#         if score > 0.5:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f"Phone {score:.2f}", (int(x1), int(y1) - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Show image
#     cv2.imshow("Phone Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

