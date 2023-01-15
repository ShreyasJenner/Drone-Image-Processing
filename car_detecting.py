#the program detects cars using a car haar cascades and morphology
#Morphology outlines the general shape of objects. If the shape of certain region in the image resembles a car then it is highlighted. 
#Error tends to occur because only shape is checked and not the pixels inside the shape

import cv2
import numpy
import glob

video = cv2.VideoWriter('results_using_morphology.mp4',cv2.VideoWriter_fourcc(*'mp4v'),15, (1904,1071))


car_cascade_src = 'cars.xml'                                                #'cars.xml' is the data set used to detect cars
car_cascade = cv2.CascadeClassifier(car_cascade_src)
print(cv2.CascadeClassifier.empty(car_cascade))                             #returns false if the 'car_cascade' has the classifier data else returns true if the data is not stored 


cnt = 0
for filename in glob.glob(r'C:\Users\HP\Desktop\Interests\Drone Data sets\Analyzing drone images\imgs\uav0000072_04488_v\*.jpg'):
    img = cv2.imread(filename)
    blur = cv2.GaussianBlur(img,(71,71),0)                                  #performs gaussian blur on the current image
    dilated = cv2.dilate(blur,numpy.ones((3,3)))                            #dilate the blurred image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))            #gets the structuring element for morphological operations
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,kernel)             #performs morphological transformations using erosion and dilation
    cars = car_cascade.detectMultiScale(closing,1.1,1)                      #detects objects of different sizes in the image. detect objects are returned as list of rectangles
    image_arr = img                                                         
    for (x,y,w,h) in cars:                                                  #to loop through the list that has detect rectangles
        cv2.rectangle(image_arr, (x,y), (x+w,y+h), (255,0,0),2)             #draws a rectange outline
        cnt +=1                                                             #counts the number of rectangles drawn that is equal to the no of cars in the image
    video.write(img)                                                        #writes the image to a video
print("cars = ", cnt)           
video.release()                                                             #stores the video in same file as the code file
