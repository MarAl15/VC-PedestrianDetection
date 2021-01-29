#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################################
#                                                   #
#                     FUNCIONES                     #
#           Mª del Mar Alguacil Camarero            #
#                                                   #
#####################################################

import cv2
import numpy as np
from matplotlib import pyplot as plt

# REPRESENTACIÓN DE LAS IMÁGENES
    # images: lista de las imágenes leídas
    # titles: nombres asociados a images
    # row: número de filas que queremos
    # col: número de columnas que deseamos
    # color: True si es una imagen a color, False en caso contrario
    # RESTRICCIÓN: debe haber igual número de imágenes que de títulos y row*col debe ser también igual a este.
def visualization(images, titles, row, col, color = [False], plot=False):
    sizeI = len(images) # Tamaño de la lista de imágenes que queremos visualize
    
    # Comprobamos que los datos introducidos son correctos
    assert sizeI==row*col and sizeI==len(titles)
        
    for i in range(sizeI):
        plt.subplot(row, col, i+1)
        
        if not plot:
            if not color[i]: # Imagen a escala de grises
                plt.imshow(images[i], "gray")
            else: # Imagen a color
                imagen = np.uint8(images[i])
                rgb_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                
                plt.imshow(rgb_img)
            
            # Ocultamos los valores de los ejes 
            plt.xticks([]), plt.yticks([])
        else:
            bins = range(1,len(images[i])+1)
            plt.bar(bins, images[i])
            
#            labels = np.array(bins).dot(20)
#            print(labels)
#            plt.xticks(bins, labels), plt.yticks([])
            plt.xticks(bins)#, plt.yticks([])
            
        
        # Asignamos un título a la imagen
        plt.title(titles[i])
    
    # Mostramos la composición de imágenes resultante
    plt.show()
    
    # Punto de parada
    cv2.waitKey(0) 
   
    
#-----------------------------------------------------------------------------------------------------------
    
#CONVOLUCION
    
# Cambiamos el intervalo de los colores
# [a,b] -> [0,255]
def cambio(x, a, b):
    return round(255*(x-a)/(b-a))

def rango(matrix):
    minimo = min(map(min,matrix))
    maximo = max(map(max,matrix))
    
#    for x in matrix:
#        x = cambio(x, minimo, maximo)
#        print(x)
    if minimo<0 or maximo>255:
        # [minimo, maximo] -> [0,255]
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i,j] = cambio(matrix[i,j], minimo, maximo)
    else: # Redondeamos sólo
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i,j] = int(round(matrix[i,j]))


def rangoColor(matrix, color=False):
    if not color:
        # Cambiamos el rango de la escala de grises
        rango(matrix)
    else:
        # Obtenemos las tres matrices de colores
        b,g,r = cv2.split(matrix)
        
        # Cambiamos el rango de cada color
        rango(b)
        rango(g)
        rango(r)
        
        # Mezclamos los colores
        matrix = cv2.merge([b,g,r])
    
    return matrix
        

def convolucion(image, kernelX, kernelY, border_type=cv2.BORDER_DEFAULT, change=True):
    row = image.shape[0] # Número de filas de la imagen
    col = image.shape[1] # Número de columnas

    blur = 1.0*image.copy() # Copiamos la imagen

    # Convolución por filas 
    for i in range(row):
        v = cv2.filter2D(src=blur[i,:], ddepth=-1, kernel=kernelX, borderType=border_type)
        
        blur[i,:] = v[:,0]

    # Convolución por columnas
    for j in range(col):
        v = cv2.filter2D(src=blur[:,j], ddepth=-1, kernel=kernelY, borderType=border_type)
        blur[:,j] = v[:,0]
      
    # Cambiamos al intervalo [0,255]
    if(change):
        rangoColor(blur)
    
    # Devolvemos la matriz convolucionada
    return blur
    

#-----------------------------------------------------------------------------------------------------------
 
    # filename: nombre de la imagen
    # border_type: tipo de borde
    # visualize: True si queremos que se visualice la imagen resultante, False en caso contrario
    # levels: número de niveles de la pirámide Gaussiana
    # read: True si la imagen debe ser leída y False en caso contrario
def pyrGauss(filename, border_type=cv2.BORDER_DEFAULT, levels=4, read=True):
    # Imagen
    if read:
        image = cv2.imread(filename)
    else:
        image = filename
        
    # Pirámide Gaussiana
    pyrGaussian = [image.copy()]
    for i in range(1, levels+1):
        pyrGaussian.append(cv2.pyrDown(src = pyrGaussian[i - 1], borderType=border_type))
        
    # Devolvemos la pirámide
    return pyrGaussian