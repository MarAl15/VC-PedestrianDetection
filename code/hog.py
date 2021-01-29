#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#####################################################
#                                                   #
#              HISTOGRAMA DE GRADIENTES             #
#           Mª del Mar Alguacil Camarero            #
#                                                   #
#####################################################

import cv2
import numpy as np
from funciones import *
from functools import reduce
from matplotlib import pyplot as plt

## CÁLCULO DEL GRADIENTE ##
def gradiente1canale(img, border_type=cv2.BORDER_DEFAULT):
    kernel = np.matrix([-1,0,1])

    # Derivada de x
    dx = cv2.filter2D(src=1.0*img, ddepth=-1, kernel=kernel, borderType=border_type)
    
    # Derivada de y
    dy = cv2.filter2D(src=1.0*img, ddepth=-1, kernel=kernel.T, borderType=border_type)
    
    return dx, dy

# Si la imagen es a color, calculamos para cada uno de los píxeles de la imagen el gradiente para cada uno de los tres canales de color. Una vez calculado los tres canales por separado, se toma, para cada píxel, aquel canal que tiene una magnitud mayor.
def gradienteOG(img, color=True, border_type=cv2.BORDER_DEFAULT):
    if color:
		# Obtenemos las tres matrices de colores
        b,g,r = cv2.split(img)
        
		# Calculamos el gradiente para cada uno de los tres canales
        dxR, dyR  = gradiente1canale(r)
        dxG, dyG  = gradiente1canale(g)
        dxB, dyB  = gradiente1canale(b)
	
        rows, cols = dxR.shape 
		
		# Derivada parcial con respecto a x
        dx = np.empty((rows, cols))
        for i in range(rows):
            dx[i] = list(map((lambda j: max(dxR[i,j], dxG[i,j], dxB[i,j])), range(cols)))

		# Derivada parcial con respecto a y
        dy = np.empty((rows, cols))
        for i in range(rows):
            dy[i] = list(map((lambda j: max(dyR[i,j], dyG[i,j], dyB[i,j])), range(cols)))
    else:
		# Derivada con respecto a x e y
        dx, dy  = gradiente1canale(img)
			
    # Orientación del gradiente (en grados)
    orientamento = np.array( (np.arctan2(dy, dx)*180/np.pi) % 360 ) # np.array(np.arctan2(dy,dx))#, dtype=np.uint8) 
        
    # Magnitud del gradiente
    grandezza = np.sqrt(dx**2 + dy**2)
    
#    dx = rangoColor(dx)
#    dy = rangoColor(dy)
#    grandezza = rangoColor(grandezza)
#    visualization([img, dx, dy], ['Original', 'Gradiente en x', 'Gradiente en y'], 1, 3, color = [color, False, False])
#    visualization([np.array(grandezza, dtype=np.uint8)], [''], 1, 1)
    
    return orientamento, grandezza
		

## HISTOGRAMAS DE GRADIENTES ##

# Factor de asignación de cada gradiente a uno de los intervalos del histograma
# max(0, 1-(angolo - (k*grado + (k-1)*grado)/2)/ grado )
# = max(0, 1-(angolo - (2*k-1)*grado/2)/ grado )
# = max(0,1-(angolo/grado - (2*k-1)/2 ))
# = max(0,1-( angolo/grado - k+1/2 ))
def pesok(angolo, k, grado=20):
    if angolo>180:
        angolo = -(360-angolo)

    return max(0,1-abs(angolo/grado-k+0.5))

# Factor de asignación de un píxel a una celda en la dirección x e y
pesox = lambda centerx, abscissa, dx: max(0, 1-abs(abscissa-centerx)/dx)
pesoy = lambda centery, ordinate, dy: max(0, 1-abs(ordinate-centery)/dy)

# Histograma de la celda ij
#   img = imagen
#   cell = celda (i,j)
#   theta = orientaciones del gradiente
#   g = magnitudes del gradiente
#   cell_size = tamaño de la celda
#   bins = número de intervalos del histograma
def istogramma(img, theta, g, cell, cell_size=(8,8), bins=9, grado=20):
    # Centro de la celda cell
    center = (cell_size[0]*((2*cell[0]+1)/2), cell_size[1]*((2*cell[1]+1)/2))
    
    # Tenemos en cuenta sólo las celdas de alrededor
    num_cells = (img.shape[0]//cell_size[0], img.shape[1]//cell_size[1])
    cell_inizio = (max(0, cell[0]-0.5), max(0,cell[1]-0.5))
    cell_fine = (min(num_cells[0]-1, cell[0]+0.5), min(num_cells[1]-1, cell[1]+0.5))
    
    inizioI = (cell_inizio[0]*cell_size[0], cell_inizio[1]*cell_size[1])
    fineI = ((cell_fine[0]+1)*cell_size[0], (cell_fine[1]+1)*cell_size[1])

    # Construimos el histograma
    h = np.zeros(bins)
    for k in range(1, bins+1):
        for i in range(int(inizioI[0]), int(fineI[0])):
            for j in range(int(inizioI[1]), int(fineI[1])):
                magnitude = g[i,j]
                if magnitude!=0:
                    weightx = pesox(center[1], j, cell_size[1])
                    if weightx!=0:
                        weighty = pesoy(center[0], i, cell_size[0])
                        if weighty!=0:
                            h[k-1] +=  weightx * weighty \
                                        * pesok(theta[i,j], k, grado) * magnitude


#    visualization([h], [''], 1, 1, plot=True)
    return h


# Normalización del vector bloque v:
    # v' = v+sqrt(||v||^2 + eps) siendo ||.|| la
    # norma euclídea y eps una constante con un valor muy pequeño
    # para evitar divisiones por cero.                                        
normalizzazione = lambda v, eps: v/np.sqrt(v.dot(v)+eps)

# Utilizaremos el gradiente sin signo
#   cell_size = tamaño de las celdas
#   block_size = tamaño del bloque
#   bins = número de intervalos del histograma
def hog(filename, cell_size=(8,8), block_size=(2,2), bins=9, eps=10**(-25), read=True):
    if read:
        # Leemos la imagen
        img = cv2.imread(filename)
    else:
        img = filename
    
    # Calculamos la orientación y la magnitud de la imagen
    orientation, magnitude = gradienteOG(img)
    
    # Dividimos la imágen en un número fijo de celdas y para cada una
    # de estas se obtiene un histograma de las orientaciones de los 
    # gradientes de esa celda.
    # Se calculan los histogramas para todas las celdas
    grado = 180//bins 
    numCells = (img.shape[0]//cell_size[0], img.shape[1]//cell_size[1])
    
    histograms = np.ndarray(numCells, dtype='O')
    for i in range(numCells[0]):
        for j in range(numCells[1]):
            histograms[i,j] = istogramma(img, orientation, magnitude, (i,j),
                                          cell_size, bins, grado)
    
    # Todos los histogramas anteriores se combinan para obtener la 
    # representación global de toda la imagen en forma de vector de 
    # características  
    
    	# Para cada bloque vamos a coger los histogramas de cada una 
    	# de las celdas y concatenarlos para obtener el vector con la
    	# representación del bloque. La normalización de los histogramas
    	# va a ser realizada a nivel de bloque, es decir, vamos a 
    	# normalizar este bloque resultado de concatenar todas las celdas.
    	
    # Solapamiento de bloque de 3/4
    numBlocks = (numCells[0]-block_size[0]+1, numCells[1]-block_size[1]+1)

    # Número de celdas/bloque
    numCB = block_size[0]*block_size[1]
    
    # Vector solución
    lim = numCB*bins
    hog = np.ndarray(numBlocks[0]*numBlocks[1]*lim, dtype='O')
    
    # Histograma de gradientes
    s = 0
    for i in range(numBlocks[0]):
        row = min(numBlocks[0]+1, i+block_size[0])
        for j in range(numBlocks[1]):
            col = min(numBlocks[1]+1, j+block_size[1])
            
            # Concatenemos los histogramas del mismo bloque
            v = np.concatenate(np.concatenate(histograms[i:row, j:col].tolist()))
            
            # Agregamos los nuevos histogramas normalizados al vector solución
            hog[s:s+lim] =  normalizzazione(v, eps)
            s += lim

#    visualization([hog], [''], 1, 1, plot=True)
    return hog