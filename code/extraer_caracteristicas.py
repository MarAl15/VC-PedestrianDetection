#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#####################################################
#                                                   #
#              EXTRAER CARACTERÍSTICAS              #
#           Mª del Mar Alguacil Camarero            #
#                                                   #
#####################################################

import os
import sys
from hog import hog 


# Extraemos las características de las distintas imágenes
def main(argv,):
    ruta = './Train/'
    
    # Nombres de todas las imágenes que contienen peatones
    nomi_pos = os.listdir(ruta+'pos')
        
    # Nombres de todas las imágenes que contienen fondo u otros objetos
    nomi_neg = os.listdir(ruta+'neg')
        
    # Abrimos los distintos archivos para guardar las características 
    # de las distintas imágenes
    pos = open("./Annotations/pos.txt", "w") 
    i = 1
    for nome in nomi_pos:#[:1000]:
        print(i, " Extrayendo caracteristicas de...", nome)
        h = hog(ruta+"pos/"+nome)
        c = ' '.join(map(str, h))
        pos.write(c+"\n")
        i+=1
    pos.close()
        
    print()
    print("------------------------------------------------------------")
    print()        

    neg = open("./Annotations/neg.txt", "w") 
    i = 1
    for nome in nomi_neg:
        print(i, "Extrayendo caracteristicas de...", nome)
        h = hog(ruta+"neg/"+nome)
        c = ' '.join(map(str, h))
        neg.write(c+"\n")
        i+=1
    neg.close()

    pass

if __name__ == "__main__":
    main(sys.argv)