# opencv kütüphanesini içe aktaralım
import cv2

# numpy kütüphanesini içe aktaralım
import numpy as np

# resmi siyah beyaz olarak içe aktaralım
image = cv2.imread("odev2.jpg",0)

# resmi çizdirelim
cv2.imshow('Odev-2',image)

# resim üzerinde bulunan kenarları tespit edelim ve görselleştirelim
edges = cv2.Canny(image = image, threshold1 = 200, threshold2 = 255)
cv2.imshow('Kenar Tespiti',edges)

# yüz tespiti için gerekli haar cascade'i içe aktaralım
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# yüz tespiti yapıp sonuçları görselleştirelim
face_rect = face_cascade.detectMultiScale(image)
for (x,y,w,h) in face_rect:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),10)
cv2.imshow("Yuz Tespiti", image)

# initialize the HOG insan tespiti algoritmamızı çağıralım ve svm'i set edelim
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# resme insan tespiti algoritmamızı uygulayalım ve görselleştirelim
(rects, weights) = hog.detectMultiScale(image, padding=(8, 8), scale=1.05)

for (xA, yA, xB, yB) in rects:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 0, 255), 2)
	
cv2.imshow("insan Tespiti", image)
