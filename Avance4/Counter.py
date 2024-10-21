
import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('autos.mp4')
#sustraccion de fondo 
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#Mejorar la imagen vinaria obtenida luego de aplicar la substraccion de fondo
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ( 3, 3))
#Declaramos el contador e igualamos a 0
car_counter = 0

while True:

    ret,frame = cap.read()
    if  ret == False: break
    #Redimensionamos el tamaño del video 
    frame = imutils.resize(frame, width=640)
    # Especificamos el area a analizar con prueba y error
    area_pts = np.array([[220, 40],[400, 40], [400, 340],[220, 340]])

    #Con ayuda de una imagen auxiliar, determinamos el area sobre la cual acuara el detector de movimiento
    imAux = np.zeros(shape=(frame.shape[:2]), dtype= np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(frame, frame, mask=imAux)

    #Aplicamos la sustraccion de fondo
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=5)

    #Encontramos los contornos presentes en fgmask, para luego basandonos en su area poder determinar si existe movimiento (autos)
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in cnts:
        if cv2.contourArea(cnt) > 800: #El valor se definio a prueba y error para determinar en base al area si es un auto
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)

            #Si el auto ha cruzado entre 220 y 380 abierto en Y, se incrementara en 1 el contador de autos
            if 220 < (y + h) < 380:
                car_counter = car_counter + 1
                cv2.line(frame, (230, 190), (390, 190), (0, 255,0), 3)

    #Visualizacion
    cv2.drawContours(frame, [area_pts], -1, (255, 0, 255), 2)
    cv2.line(frame, (230, 190), (390, 190), (0, 255, 255), 1)
    cv2.rectangle(frame, (frame.shape[1]-70, 215), (frame.shape[1]-5, 270), (0, 255, 0), 2)
    cv2.putText(frame, str(car_counter), (frame.shape[1]-55,250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),2)
    cv2.imshow('Frame', frame)
    cv2.imshow('fgmask', fgmask)

    k = cv2.waitKey(1) & 0xFF #Aqui modificamos la velociad del video 
    if k == 27: # Si presiona 'Esc', se sale del loop
        break

cap.release()
cv2.destroyAllWindows()