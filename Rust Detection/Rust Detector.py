import cv2
import matplotlib.pyplot as plt
import numpy as np

image=cv2.imread('rust-featimage-768x405.jpg.webp')
image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

lower_red=np.array([0,70,70])
upper_red=np.array([10,255,255])
mask=cv2.inRange(image,lower_red,upper_red)


lower_red=np.array([170,50,50])
upper_red=np.array([180,255,255])
mask1=cv2.inRange(image,lower_red,upper_red)
masktotal=mask1+mask

kernal=np.ones((5,5),np.uint8)
corrmask=cv2.dilate(masktotal,kernal,iterations=5)

contours,hierarchy=cv2.findContours(masktotal,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for pic,contour in enumerate(contours):
    area=cv2.contourArea(contour)
    if(area>300):
        x,y,w,h=cv2.boundingRect(contour)
        image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(image,"Rust",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        
output=cv2.bitwise_and(image,image,mask=masktotal)

plt.imshow(output)
plt.show()