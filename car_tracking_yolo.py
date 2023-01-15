#makes use of tinyyolov3 to detect cars
#modified frames with detected cars are then written to a video using OpenCV.
#video format is mp4 but can be changed based on user's need
#modules used are:  imageai to use tinyolov3
#glob to return to the program all the images in which cars are to be detected
#OpenCV to read images and create videos

from imageai.Detection import ObjectDetection
import glob
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

video = cv2.VideoWriter('results_yolo.mp4',cv2.VideoWriter_fourcc(*'mp4v'),15, (1904,1071))
detector = ObjectDetection()

input_path = r"C:\Users\HP\Desktop\Interests\drone_data_sets\analyzing_drone_images\imgs\cars.jpg"

model_path = r"C:\Users\HP\Desktop\Interests\drone_data_sets\analyzing_drone_images\yolo_method\models\yolo-tiny.h5"

detector.setModelTypeAsTinyYOLOv3()         
detector.setModelPath(model_path)
detector.loadModel()             
custom_objects = detector.CustomObjects(car=True)


for filename in glob.glob(r"C:\Users\HP\Desktop\Interests\drone_data_sets\analyzing_drone_images\imgs\uav0000071_03240n_v\*.jpg"):
    img = cv2.imread(filename)
    detection = detector.detectObjectsFromImage(input_image=filename, output_image_path=output_path,custom_objects=custom_objects)
    print(output_path)
    img = cv2.imread(output_path)
    video.write(img)
    for eachitem in detection:
        print(eachitem)

video.release()

