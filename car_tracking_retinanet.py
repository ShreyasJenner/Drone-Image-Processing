from imageai.Detection import ObjectDetection
import glob
import cv2
import numpy

video = cv2.VideoWriter('results_retinanet(1).mp4',cv2.VideoWriter_fourcc(*'mp4v'),15, (1904,1071))
detector = ObjectDetection()

output_path = r"C:\Users\HP\Desktop\Interests\drone_data_sets\analyzing_drone_images\yolo_method\output\test.jpg"

model_path = r"C:\Users\HP\Desktop\Interests\drone_data_sets\analyzing_drone_images\imageai\resnet50_coco_best_v2.1.0.h5"

detector.setModelTypeAsRetinaNet()         
detector.setModelPath(model_path)
detector.loadModel()             

custom_objects = detector.CustomObjects(car=True)
cnt = 0
for filename in glob.glob(r"C:\Users\HP\Desktop\Interests\drone_data_sets\analyzing_drone_images\imgs\uav0000071_03240_v\*.jpg"):
    img = cv2.imread(filename)
    detections = detector.detectObjectsFromImage(input_image=filename, output_image_path=output_path, custom_objects=custom_objects,display_percentage_probability=False)
    img = cv2.imread(output_path)
    video.write(img)
    for eachitem in detections:
        print(eachitem)
        cnt += 1

video.release()
print("no of car=",cnt)

