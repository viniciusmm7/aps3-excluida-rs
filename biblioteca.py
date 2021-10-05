#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

from sklearn.linear_model import LinearRegression, RANSACRegressor

def segmenta_linha_amarela(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem bgr e retornar uma máscara com os segmentos amarelos do centro da pista em branco.
        Utiliza a função cv2.morphologyEx() para limpar ruidos na imagem
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30/2, 153, 160), (70/2, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    morpho = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return morpho

def encontrar_contornos(mask):
    """Não mude ou renomeie esta função
        deve receber uma imagem preta e branca e retornar todos os contornos encontrados
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    return contours

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    x = int(x)
    y = int(y)
    cv2.line(img,(x - size,y),(x + size,y),color,2)
    cv2.line(img,(x,y - size),(x, y + size),color,2)

def encontrar_centro_dos_contornos(bgr, contornos):
    """Não mude ou renomeie esta função
        deve receber uma lista de contornos e retornar, respectivamente,
        a imagem com uma cruz no centro de cada segmento e o centro de cada. 
        formato: img, x_list, y_list
    """
    img = bgr.copy()
    x_list = []
    y_list = []

    for contorno in contornos:
        try:
            m = cv2.moments(contorno)
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]

            x_list.append(cx)
            y_list.append(cy)
            crosshair(img, (cx, cy), 5, (255, 0, 0))

        except ZeroDivisionError:
            pass

    return img, x_list, y_list


def desenhar_linha_entre_pontos(bgr, X, Y, color):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e retornar uma imagem com uma linha entre os centros EM SEQUENCIA do mais proximo.
    """
    img = bgr.copy()
    delta_x = []
    delta_y = []
    for i in range(len(X) - 1):

        pos = (int(X[i]), int(Y[i]))
        pos_next = (int(X[i + 1]), int(Y[i + 1]))
        img = cv2.line(img, pos, pos_next, color, 2)

    return img

def regressao_por_centro(bgr, x_array, y_array):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e estimar a melhor reta, utilizando o metodo preferir, que passa pelos centros. Retorne a imagem com a reta e os parametros da reta
        
        Dica: cv2.line(img,ponto1,ponto2,color,2) desenha uma linha que passe entre os pontos, mesmo que ponto1 e ponto2 não pertençam a imagem.
    """
    img = bgr.copy()
    y_array = np.array(y_array).reshape(-1, 1)
    reg = LinearRegression().fit(y_array, x_array)
    ransac = RANSACRegressor(reg).fit(y_array, x_array)
    reg = ransac.estimator_
    lm = reg.coef_
    o = reg.intercept_
    y1 = 0
    x1 = int(lm * y1 + o)
    y2 = img.shape[0]
    x2 = int(lm * y2 + o)

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img, lm

def calcular_angulo_com_vertical(img, lm):
    """Não mude ou renomeie esta função
        deve receber uma imagem contendo uma reta, além da reggressão linear e determinar o ângulo da reta com a vertical, utilizando o metodo preferir.
    """
    angulo_radianos = np.arctan(lm)
    angulo = np.degrees(angulo_radianos)
    
    return angulo

if __name__ == "__main__":
    print('Este script não deve ser usado diretamente')
