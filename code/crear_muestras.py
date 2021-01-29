#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#####################################################
#                                                   #
#              REDIMENSIONAR IMÁGENES               #
#           Mª del Mar Alguacil Camarero            #
#                                                   #
#####################################################

import cv2
import unittest
import numpy as np
from matplotlib import pyplot as plt
from funciones import *
import sys
#import os

#os.chdir('./images') # cambiamos de directorio


# REDIMENSIONAR: Redimensionamos la imagen 'filename'
def ridimensionare(filename, dim=(128, 64)):
    img  = cv2.imread(filename)
    n = img.shape[0] # Alto
    m = img.shape[1] # Ancho
    
    sobrepasa = False
    
    width = dim[1]
    percent = width/float(m)
    
    if n*percent>dim[0]:
        sobrepasa = True
        img = img.transpose((1,0,2))
        dim = (dim[1], dim[0])
        height = dim[1]
        percent = height/float(n)
        
    # Redimensionamos la foto manteniendo las proporciones
    img = cv2.resize(img, None, fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
            
    # Rellenamos para que tenga las dimensiones deseadas
    res = np.empty((dim[0], dim[1], 3) )
    differenza = dim[0] - img.shape[0]
    i = differenza//2
    height = img.shape[0]
    
    res[0:i] = img[0]
    res[i:i+height] = img[:height]
    res[i+height:] = img[height-1]
    
    # Volvemos a ponerla bien
    if sobrepasa:
        res = res.transpose((1,0,2))

            
    # Guardamos la imagen
    cv2.imwrite(filename, res)
    
def recortar(filename, xmin, ymin, xmax, ymax, num):
    img  = cv2.imread(filename)
    
    filename = filename.lower()
    nome = filename[:len(filename)-4]+"_"+str(num)+filename[len(filename)-4:]
    cv2.imwrite(nome, img[ymin-8:ymax+8, xmin-8:xmax+8])
    
    return nome
    
def leggereNumero(linea, inizio):
    numero = linea[inizio]
    
    if len(linea)>inizio:
        fin = inizio+1
        letra = linea[fin]
    
    while letra!=',' and letra!=')' and len(linea)>fin:
        numero += letra
        fin += 1
        letra = linea[fin]
    
    return int(numero), fin
    
def main(argv,):
    fileI = open('Train/pos.lst')
    imgs = []
    for immagine in fileI.readlines(): 
        print("Leyendo...", immagine)
        im = str("./"+immagine)
        im = im[:len(im)-1]
        imgs.append(im)

#        ridimensionare(im)
#    
#    file.close()
#    ridimensionare('./Train/pos/crop_000606.png')
    
    # Recortamos por donde se encuentra la persona
    file = open('Train/annotations.lst')
    numI = 0
    for annotazione in file:
        nota = annotazione[:len(annotazione)-1]
        num = 1
        fileA = open(nota)
        print("->Leyendo...", nota)
        for linea in fileA.readlines():
            testo =  'Bounding box for object '+str(num)+' "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : ('
            inizio = len(testo)
            xmin, inizio = leggereNumero(linea, inizio)
            ymin, inizio = leggereNumero(linea, inizio+2)
            
            xmax, inizio = leggereNumero(linea, inizio+5)
            ymax, inizio = leggereNumero(linea, inizio+2)
            
            nome = recortar(imgs[numI], xmin, ymin, xmax, ymax, num)
            ridimensionare(nome, dim=(160, 96))
            num+=1
        numI+=1
    pass

if __name__ == "__main__":
    main(sys.argv)