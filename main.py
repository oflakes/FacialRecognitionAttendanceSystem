# Testing OpenCV

import cv2
import numpy as np
import face_recognition

imgWW = face_recognition.load_image_file(
    'Images/Walter White.jpg')
imgWW = cv2.cvtColor(imgWW, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/ww2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgWW)[0]
encodeWW = face_recognition.face_encodings(imgWW)[0]
cv2.rectangle(imgWW, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeWW], encodeTest)
faceDis = face_recognition.face_distance([encodeWW], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Walter White', imgWW)
cv2.imshow('Walter Test', imgTest)
cv2.waitKey(0)
