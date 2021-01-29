#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#####################################################
#                                                   #
#                TRAIN(SVM) - TEST                  #
#           Mª del Mar Alguacil Camarero            #
#                                                   #
#####################################################

import sys
import pickle
import numpy as np
from sklearn.svm import SVR, LinearSVC

# Cargamos los vectores de características del fichero pasado como argumento
def leggere(filename):
    file = open(filename, "r")
    
    # Leemos cada línea del fichero (vector de característica de cada imagen) 
    # y la transformamos en un vector de reales
    linee = file.readlines()
    matrice = []
    for linea in linee:
        matrice.append( np.array([[m.strip() for m in n] 
                                for n in [linea.split(" ")]][0],
                                dtype=float) )                   
        # for linea in linee]][0],
                                
    
    # Convertimos la lista en una matriz
    N = len(matrice) # Número de muestras 
    M = len(matrice[0]) # Tamaño del vector de características
    matrice = np.array(matrice).reshape(N, M)
    
    return N, matrice
    

# Entramiento con SVM. Guardamos el modelo resultante
def allenare(matricePositiva, matriceNegativa, lineal=True, C=0.01):
    Npositive = len(matricePositiva) # Número de muestras positivas
    Nnegative = len(matriceNegativa) # Número de muestras negativas
    
    # Inicializamos la estructura que pasaremos al modelo para entrenar
    # X -> muestras a entrenar
    # y -> etiquetas (+1 si en la imagen se encuentra un peatón 
    #                  y -1 en caso contrario )
    X = np.concatenate((matricePositiva, matriceNegativa), axis=0)
    y = np.append(np.ones(shape=(1, Npositive)),
                  -1*np.ones(shape=(1, Nnegative)))
    
    if lineal: # SVM lineal 
    # dual : bool, (default=True)->Select the algorithm to either solve the dual 
    # or primal optimization problem. Prefer dual=False when n_samples > n_features.
        M = len(matricePositiva[0]) # Tamaño del vector de características
        if (Npositive+Nnegative)>M:
            dual = False
        else:
            dual = True
        model = LinearSVC(C = C, dual = dual)
    else: # SVM con kernel gaussiano 
        model = SVR(kernel = 'rbf', C = C, gamma = 3e-2)   
    
    model.fit(X, y)
    
    # Porcentaje de acierto para el conjunto de entrenamiento
    print("Train")
    successo(matricePositiva, matriceNegativa, model)
    
    # Guardamos el modelo
    modello = open("./Annotations/model.txt", "wb")
    pickle.dump(model, modello)
    
    return model

# Prueba
def successo(matricePositiva, matriceNegativa, model):
    erroriPos = 0
    erroriNeg = 0
    
    Npositive = len(matricePositiva) # Número de muestras positivas
    Nnegative = len(matriceNegativa) # Número de muestras negativas
    
    for vettore in matricePositiva:
        if model.predict([vettore])<=0:
            erroriPos += 1
    
    for vettore in matriceNegativa:
        if model.predict([vettore])>=0:
            erroriNeg += 1
    
    # Porcentaje de acierto para el conjunto dado  
    print("TPR:", 1-erroriPos/(Npositive + Nnegative))
    print("TNR:", 1-erroriNeg/(Npositive + Nnegative))
    print("Porcentaje de acierto:", 1-(erroriPos+erroriNeg)/(Npositive + Nnegative))
    
    
def main(argv,):
    ## Cargamos las características extraídas de las distintas imágenes ##
    # Imágenes con peatones
        # Npositive: Número de muestras positivas
        # M: Tamaño del vector de características
        # matricePositiva: vector de características de cada muestra con peatones
    Npositive, matricePositiva = leggere("./Annotations/pos.txt")
    
    # Imágenes con fondo y otros objetos 
        # Nnegative: Número de muestras negaticas
        # M: Tamaño del vector de características
        # matriceNegativa: vector de características de cada muestra sin peatones
    Nnegative, matriceNegativa = leggere("./Annotations/neg.txt")
    
    
    ## ENTRENAMIENTO ##
    # Mezclamos las matrices de forma aleatoria (Las matrices multidimensionales
    # sólo se mezclan a lo largo del primer eje)
    np.random.shuffle(matricePositiva)
    np.random.shuffle(matriceNegativa)
    # Entrenamos SVM con el (percentuale*100)% de muestras positivas y de negativas
    percentuale = 1
    S = int(percentuale*Npositive)
    T = int(percentuale*Nnegative)
    model = allenare(matricePositiva[:S], matriceNegativa[:T])
    
    ## TEST ##
    if S!=Npositive:
        print()
        print("Test")
        successo(matricePositiva[S:Npositive], matriceNegativa[T:Nnegative], model)
    
    pass

if __name__ == "__main__":
    main(sys.argv)
