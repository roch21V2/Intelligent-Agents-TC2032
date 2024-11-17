from simpleai.search import SearchProblem, astar, greedy, depth_first, uniform_cost
from simpleai.search.viewers import BaseViewer, ConsoleViewer, WebViewer
import numpy as np
import time 


start_time = time.time()
scale = 10
mapa = np.load('mars_map1.npy')

nr,nc = mapa.shape

class Rover(SearchProblem):
    def __init__(self,Initial_state, Final_state):
        SearchProblem.__init__(self,Initial_state)
        x2,y2 = Final_state
        self.goal_state = (x2,y2)

    def actions(self,state):
        x2,y2 = state
        self.x = nr - round(y2/scale)
        self.y = round(x2/scale)
        self.z = mapa[self.x][self.y]

        actions=[]
        vecinos = [(self.x +1, self.y), (self.x - 1, self.y ), (self.x, self.y -1), (self.x, self.y + 1), (self.x + 1,self.y + 1 ), (self.x +1, self.y -1),(self.x - 1, self.y -1), (self.x - 1, self.y + 1)]
        for x,y in vecinos:
            if(mapa[x][y]!= -1 and abs(self.z - mapa[x][y]) <= 0.25):
                actions.append(((self.x - x)*scale, (self.y - y)*scale))

    
        return actions
    
    def result(self, state, action):
        x,y = state
        x2,y2 =action
        new_state = x+x2,y+y2
        return(tuple(new_state))
    

    def heuristic(self, state):
        x,y = state
        x2,y2 = self.goal_state
        x2 = nr - round(y2/scale)
        y2 = round(x2/scale)

        x = nr - round(y/scale)
        y = round(x/scale)
        return(np.sqrt((x2 - x)**2+ (y2-y)**2+ (mapa[x2][y2]-mapa[x][y])**2))
    

        
    def display(result):
        if result is not None:
            for i, (action, state) in enumerate(result.path()):
                if action == None:
                    print('Inicio')
                elif i == len(result.path()) - 1:
                    print(i,'Después', action)
                    print('Meta lograda con un costo de', result.cost)
                else:
                    print(i,'- ', action)

                print('  ', state)
        else:
            print('Mala configuración del problema')


    def cost(self, state, action, state2):
        return 1
    def is_goal(self, state):
        return state==self.goal_state
    

result = greedy(Rover((4978,10628),(5300,10628)),graph_search=True)
#result = astar(Rover((4978,10628),(5300,10628)),graph_search=True)
#result = depth_first(Rover((4978,10628),(5300,10628)),graph_search=True)
#result = uniform_cost(Rover((4978,10628),(5300,10628)),graph_search=True)




end_time = time.time()
elapsed_time = end_time - start_time


for i, (action, state) in enumerate(result.path()):
    print()
    if action == None:
        print('Initial configuration')
    elif i == len(result.path()) - 1:
        print('Despues de moverse', action, ' metros. Goal achieved!')
        print('Meta lograda con un costo de', result.cost)
    else:
        print('Despues de moverse', action, 'metros en la coordenada real')

    for item in state:
        print("{:2}".format(item), end = " ")
    print()

print("Elapsed time: ", elapsed_time, " seconds")
