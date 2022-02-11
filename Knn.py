import numpy as np
from collections import Counter
import utils




class KNN:
    #Revisa entre los 3(k) nodos mas cercanos y la mayoria indica en que categoria cae el nuevo x
    def __init__(self,k=3):
        self.k = k
        
    def fit(self,X,y):
        # No entrena , guarda las X para el futuro
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
        
    #Metodo para predecir cada x en el set X
    def _predict(self,x):
        #Calcular distancias
        distances = [utils.euclidean_distance(x,x_train) for x_train in self.X_train]
        
        # Obtener las categorias de los k-cercanos 
        
        # Ordena y da los indices , y solo saco los k(cercanos)
        k_indices = np.argsort(distances)[:self.k]
        
        # En y tengo las etiquetas , con los indices anteriores elijo las categorias mas cercanas
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        #Mayoria gana , y elijo la categoria que mas se repite
        
        #Cuento las categorias y el elijo el primer mas comun
        
        #Counter.most_commen() devuelve (Objecto que mas se repite,veces que se repite)
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]