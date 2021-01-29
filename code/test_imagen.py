#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#####################################################
#                                                   #
#                  TEST  IMAGEN                     #
#           Mª del Mar Alguacil Camarero            #
#                                                   #
#####################################################

import cv2
import sys
import pickle
import numpy as np
from hog import hog
from funciones import *

# Detectar peatón en la imágen si lo hubiese
    # filename: nombre de la imágen 
    # finestra: tamaño de la ventana
    # salto: número de píxeles que vamos a saltar para deslizar la ventana
    #por toda la imágen
    # n: niveles de la pirámide Gaussiana
    # soglia = umbral para determinar si la predicción es correcta
    # level: primer nivel a considerar
    # lineal: True si hemos SVM lineal y False en caso contrario
def scoprirePedone(filename, finestra=(128, 64), salto=(64, 32), n=3, soglia=0, 
                   level=0, lineal=True):
    # Cargamos el modelo entrenado
    with open("./Annotations/model.txt","rb") as fd:
        model=pickle.load(fd)
        
    # Pirámide Gaussiana de n niveles
    piramide = pyrGauss(filename, levels=n)
    imagen = piramide[0]
 
    for scala, img in enumerate(piramide[level:]):
        (N, M, s) = img.shape
#        im = img.copy()
        for i in range(0, N, salto[0]):
            for j in range(0, M, salto[1]):
                if (i+finestra[0])>N and (j+finestra[1])<=M \
                        and (i+finestra[0]-N)<5: 
                    res = np.empty((finestra[0], finestra[1], 3) )
                    n = N-i
                    w = 0#differenza//2
                    res[w:w+n] = img[i:i+finestra[0], j:j+finestra[1]]
                    res[w+N:] = img[-1, j:j+finestra[1]]
                else:
                    res = img[i:i+finestra[0], j:j+finestra[1]]
                
                if res.shape[0]==finestra[0] and res.shape[1]==finestra[1]:
#                    cv2.rectangle(im,(j,i), (j+finestra[1], i+finestra[0]),
#                                               (255,0,0), 2)
#                    visualization([im], [''], 1, 1, color=[True])
 
                    # Hallamos el vector de características de la subimagen
                    vector = hog(res, read=False)
                    
                    # Obtenemos la predicción de las puntaciones de confianza 
                    # para las muestras
                    if lineal:
                        decision_func = model.decision_function([vector])
                    else:
                        decision_func = model.predict([vector])
                        
                    # Si es mayor un umbral prefijado, la consideramos
                    if decision_func>=soglia:
#                        print(decision_func)
                        s = 2**(scala+level)
                        cv2.rectangle(imagen,(s*j,s*i), (s*(j+finestra[1]), 
                                              s*(i+finestra[0])),
                                              (0,255,0), 2)
                        
    visualization([imagen], [''], 1, 1, color=[True])            

def main(argv,):
    immagine = "./images/prueba.png"    
    scoprirePedone(immagine, level=1)  

    pass

if __name__ == "__main__":
    main(sys.argv)
    
