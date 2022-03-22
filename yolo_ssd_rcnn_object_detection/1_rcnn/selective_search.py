import cv2
import random

image = cv2.imread("pyramid.jpg")
image = cv2.resize(image, dsize = (600,600))
cv2.imshow("image",image)

# ilklendir ss
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()

print("start")
rects = ss.process()

output = image.copy()

for (x,y,w,h) in rects[:50]:
    color = [random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(output, (x,y),(x+w,y+h),color,2)
    
cv2.imshow("output",output)