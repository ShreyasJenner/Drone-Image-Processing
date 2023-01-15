import cv2
import numpy as np

video = cv2.VideoWriter('MIL_tracker.mp4',cv2.VideoWriter_fourcc(*'mp4v'),15, (1904,1071))
cap = cv2.VideoCapture(r"C:\Users\HP\Desktop\Interests\drone_data_sets\analyzing_drone_images\results_retinanet(1).mp4")

ret, frame = cap.read()

bbox = cv2.selectROI(frame, False)
tracker = cv2.TrackerMIL_create()
tracker.init(frame, bbox)

trajectory = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        center = (x + w/2, y + h/2)
        trajectory.append(center)
        video.write(frame)

    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

video.release()

cv2.destroyAllWindows()

print(trajectory)
