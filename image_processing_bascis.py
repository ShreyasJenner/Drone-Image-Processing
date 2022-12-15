import cv2,time,numpy

image = cv2.imread(r'C:\Users\HP\Desktop\Interests\Drone Data sets\Analyzing drone images\imgs\road_1.jpg',0)

print(image)
print(image.shape)
print(image.ndim)

#cv2.imshow('china',image)
#cv2.waitKey(0)

blur = cv2.GaussianBlur(image,(59,59),0)
#cv2.imshow('blurred',blur)
#cv2.waitKey(0)

dilated = cv2.dilate(blur,numpy.ones((3,3)))
cv2.imshow('dilate',dilated)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('closing',closing)
#cv2.waitKey(0)

#car_cascade_src = 'cars.xml'
#car_cascade = cv2.CascadeClassifier(car_cascade_src)
#cars = car_cascade.detectMultiScale(closing,1.1,1)
#print("cars=",cars)

#cnt = 0
#image_arr = image
#for (x,y,w,h) in cars:
#    cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
#    cnt +=1
#print(cnt, "cars found")
#image_arr = cv2.resize(image_arr,(800,600))
#cv2.imshow('circled',image_arr)
#cv2.waitKey(0)

