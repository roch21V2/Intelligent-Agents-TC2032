import numpy as np
from simpleai.search import SearchProblem,breadth_first,depth_first,astar,uniform_cost,greedy
import math
from simpleai.search.viewers import BaseViewer, ConsoleViewer, WebViewer
from timeit import default_timer as timer
import random
#---------------------------------------------------------------------------------------------------------------
#   Definicion de problema
#---------------------------------------------------------------------------------------------------------------



class Crateres(SearchProblem):

    def _init_(self,mapa,escala,estadoInicial,estadoMeta,nr,nc):

        self.mars_map = mapa
        self.escala =  escala

        r_0 = nr - round(estadoInicial[1] / escala)
        c_0 = round(estadoInicial[0]/escala)


        SearchProblem._init_(self,(r_0,c_0))

        r_1 = nr - round(estadoMeta[1] / escala)
        c_1 = round(estadoMeta[0]/escala)

        self.goal_state =  (r_1,c_1)

        self.r = r_0
        self.c = c_0
        self.z = mapa[self.r][self.c]
        self.alpha = 0.99
        self.T = 1000

        
    def actions(self,state):
        """
        Regresa todas las posibles acciones, donde tambien evalua si puede avanzar a su coordenada vecina
        """
        r = state[0] #renglon actual
        c = state[1] # columna actual
        z = self.mars_map[r][c] #altura actual

        vecinos = [(r + 1,c),(r - 1,c),(r,c+1),(r,c-1),(r+1,c+1),(r-1,c-1),(r+1,c-1),(r-1,c+1)]

        #Todas las acciones que el robot si puede ejecutar
        acciones = []
        for vecino in vecinos:
            r_vecino = vecino[0]
            c_vecino = vecino[1]
            z_vecino = self.mars_map[r_vecino][c_vecino]         
            dif = abs(z -  z_vecino)

            if (z_vecino != -1) and (dif <= 2):
                acciones.append((r_vecino,c_vecino))
            else:
                continue
        
        return acciones

        
    
    def result(self,state,action):
        return action
        


        
    
    def is_goal(self,state):
        return state == self.goal_state
    
    def cost(self,state,action,state2):
        dif_z = abs(self.mars_map[state[0]][state[1]] - self.mars_map[state2[0]][state2[1]]) 
        return dif_z
    
    def heuristic(self, state):
        r = state[0] #renglon actual
        c = state[1] # columna actual

        r_final = self.goal_state[0]
        c_final = self.goal_state[1]

        dist_r = abs(r_final - r)
        dist_c = abs(c_final - c) 
        
        dist_euclediana = math.sqrt(dist_r * 2 + dist_c * 2)

        return dist_euclediana
    
    def recocidoSimulado(self):
        step = 0
        while self.T > 1:
            
            print(f"Iteracion {step} ---> renglon: {self.r}, columna: {self.c}, altura: {self.z}")
            step += 1
            vecinos = [(self.r + 1,self.c),(self.r - 1,self.c),(self.r,self.c+1),(self.r,self.c-1),(self.r+1,self.c+1),(self.r-1,self.c-1),(self.r+1,self.c-1),(self.r-1,self.c+1)]
            posicion =  random.randint(0,len(vecinos) - 1)
            vecino = vecinos[posicion]
            
            z_vecino = self.mars_map[vecino[0]][vecino[1]]         
            dif = abs(self.z - z_vecino)     
            
            if (z_vecino < self.z) and (dif <= 2.0):# Si es menor la altura, los paramtros cambian
                self.r = vecino[0]
                self.c = vecino[1]
                self.z = z_vecino
                
            else:
                #Si no es menor , entonces entonces las variables (a,b y c) son aceptadas con una probabilidad que 
                #depende de la temperatura y la diferencia entre el nuevo costo y el costo actual. Esta probabilidad es calculada utilizando la distribución de Boltzmann
                p = math.exp(-(z_vecino - self.z) / self.T) 
                if random.random() < p:
                    self.r = vecino[0]
                    self.c = vecino[1]
                    self.z = z_vecino
            self.T *= self.alpha #actualiza el valor de la temperatura (decrementa)
    


# Despliega los resultados
def display(result):
    if result is not None:
        for i, (action, state) in enumerate(result.path()):
            z = mars_map[state[0]][state[1]]
            if action == None:
                print('Configuración inicial')
                print("\nAltura inicial = ",z," metros")
            elif i == len(result.path()) - 1:
                print(i,'- Coordenada (columna,renglon)', action)
                print("\nAltura actual = ",z," metros")
                print('\n¡Meta lograda con costo =', result.cost,' metros')
                print("\nMeta lograda")
                
            else:
                print(i,'- Después de seleccionar coordenada (renglon y columna de la matriz): ', action)
                print("\nAltura actual = ",z," metros")

            print()
            print()
    else:
        print('Mala configuración del problema')


#---------------------------------------------------------------------------------------------------------------
#   Programa
#---------------------------------------------------------------------------------------------------------------

my_viewer = None
#my_viewer = None     # Solo estadísticas
#my_viewer = ConsoleViewer()    # Texto en la consola
#my_viewer = WebViewer()        # Abrir en un browser en la liga http://localhost:8000

# Crea un PSA y lo resuelve con la búsqueda primero en anchura
#result = breadth_first(CleanupPuzzle(11), graph_search=True, viewer=None)


print("Metodo de busqueda greedy\n")
mars_map =  np.load('mars_map2.npy')
nr,nc = mars_map.shape #guarda el numero de filas y columnas de la matriz

ruta1 = Crateres(mapa=mars_map,escala=10.045,estadoInicial = (3350,5800),estadoMeta=(3555,5323),nr=nr,nc=nc) #cercano
ruta2 = Crateres(mapa=mars_map,escala=10.045,estadoInicial = (3350,5800),estadoMeta=(3495,5434),nr=nr,nc=nc) #cercano
ruta3 = Crateres(mapa=mars_map,escala=10.045,estadoInicial = (3350,5800),estadoMeta=(2511,5464),nr=nr,nc=nc) #posicion alejada
ruta4 = Crateres(mapa=mars_map,escala=10.045,estadoInicial = (3350,5800),estadoMeta=(2641,5163),nr=nr,nc=nc)#posicion alejada
ruta5 = Crateres(mapa=mars_map,escala=10.045,estadoInicial = (3350,5800),estadoMeta=(2913,4912),nr=nr,nc=nc) #posicion mas alejada

start1 = timer()
result1_greedy = greedy(ruta1,graph_search=True,viewer=my_viewer)
end1 =  timer()

start2 = timer()
result2_greedy = greedy(ruta2,graph_search=True,viewer=my_viewer)
end2 =  timer()


start3 = timer()
result3_greedy = greedy(ruta3,graph_search=True,viewer=my_viewer)
end3 =  timer()

start4 = timer()
result4_greedy = greedy(ruta4,graph_search=True,viewer=my_viewer)
end4 =  timer()


start5 = timer()
result5_greedy = greedy(ruta5,graph_search=True,viewer=my_viewer)
end5 =  timer()


print("\nMétodo de búsqueda primero por profundidad\n")
display(result1_greedy)
print("\n-Tiempo de ejecucion (segundos): ",round(end1 - start1,8))

print("\nMétodo de búsqueda primero por profundidad\n")
display(result2_greedy)
print("\n-Tiempo de ejecucion (segundos): ",round(end2 - start2,8))

print("\nMétodo de búsqueda primero por profundidad\n")
display(result3_greedy)
print("\n-Tiempo de ejecucion (segundos): ",round(end3 - start3,8))

print("\nMétodo de búsqueda primero por profundidad\n")
display(result4_greedy)
print("\n-Tiempo de ejecucion (segundos): ",round(end4 - start4,8))

print("\nMétodo de búsqueda primero por profundidad\n")
display(result5_greedy)
print("\n-Tiempo de ejecucion (segundos): ",round(end5 - start5,8))

print("Metodo de busqueda Recocido simulado\n")

#ruta1_RS = Crateres(mapa=mars_map,escala=10.045,estadoInicial = (3350,5800),estadoMeta=(3555,5323),nr=nr,nc=nc) #cercano
#ruta2_RS = Crateres(mapa=mars_map,escala=10.045,estadoInicial = (3350,5800),estadoMeta=(3495,5434),nr=nr,nc=nc) #cercano
#ruta3_RS = Crateres(mapa=mars_map,escala=10.045,estadoInicial = (3350,5800),estadoMeta=(2511,5464),nr=nr,nc=nc) #posicion alejada
#ruta4_RS = Crateres(mapa=mars_map,escala=10.045,estadoInicial = (3350,5800),estadoMeta=(2641,5163),nr=nr,nc=nc)#posicion alejada
ruta5_RS = Crateres(mapa=mars_map,escala=10.045,estadoInicial = (3350,5800),estadoMeta=(2913,4912),nr=nr,nc=nc) #posicion mas alejada


#ruta1_RS.recocidoSimulado()
#ruta2_RS.recocidoSimulado()
#ruta3_RS.recocidoSimulado()
#ruta4_RS.recocidoSimulado()
ruta5_RS.recocidoSimulado()