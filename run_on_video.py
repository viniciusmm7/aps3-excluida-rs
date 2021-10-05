#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import biblioteca

print("Baixe o arquivo a seguir para funcionar: ")
print("https://github.com/Insper/robot202/raw/master/projeto/centro_massa/video.mp4")

cap = cv2.VideoCapture('line_following.mp4')

while(True):
    # Capture frame-by-frameqq
    ret, img = cap.read()
    # frame = cv2.imread("frame0000.jpg")
    # ret = True
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")
        break
    else:
        bgr = img.copy()
        mask = biblioteca.segmenta_linha_amarela(bgr)
        contours = biblioteca.encontrar_contornos(mask)
        img, X, Y = biblioteca.encontrar_centro_dos_contornos(img, contours)
        img, lm = biblioteca.regressao_por_centro(img, X, Y)
        angulo = biblioteca.calcular_angulo_com_vertical(img, lm)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(angulo.round(1)[0]), (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Imagem original
        cv2.imshow('img', img)
        # Mascara
        cv2.imshow('mask', mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()