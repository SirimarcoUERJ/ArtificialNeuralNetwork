import numpy as np
import random
import pickle as pkl
from time import sleep

# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class perceptron:

    def __init__(self, lines = 3, columns = 3, n_individuals = 1, layer = [9, 18, 18, 9]):
        self.lines = lines 
        self.columns = columns
        self.layer = layer
        self.bias = 1
        self.lear_rate = 0.1
        self.mutation_rate = 0.01
        self.name = self.file_checker(n_individuals)[1]
        if self.name == []:
            self.name = self.file_checker(n_individuals)[1]    
    
    def weights(self, name):
        ''' Esse método não recebe nada, ela usa a biblioteca random para pegar numeros aleatorios para usar como pesos da rede neural.
            abre um arquivo, usando a biblioteca pickle, ele será usado pela função mutation() para gerar os novos pesos.
            imput => none
            output => none
        '''
        
        weights = []
        for i in range(len(self.layer) - 1):
            weights.append([])

        for n in range(len(self.layer) - 1):

            for i in range(self.layer[n + 1]):
                weight = []
            
                for j in range(self.layer[n] + 1):
                    weight.append(random.randrange(-100,100,1)/100)
                weights[n].append(weight)
    
        with open(name, "wb") as file:
            pkl.dump((weights, self.layer), file)

        # print(weights)
        return None

    def crossover(self, fist_individual, second_individual, third_individual):
        """ atualiza os pesos das rede neural que não sobrevivem ao desafio, gerando novos pesos usando uma mistura de pesos das redes neurais que sobreviveram.
            imput   => nome1, nome2, nome3
            output  => None
        """

        individuals = [fist_individual, second_individual]
        old_weights = []

        with open(third_individual, "rb") as file:
            weights, number_layers = pkl.load(file)
        
        new_weights = weights

        for i in range(2):
            
            with open(individuals[i], "rb") as file:
                weights, number_layers = pkl.load(file)
            old_weights.append(weights)

        # print(third_individual+" = ", new_weights, "\n\n",fist_individual+" = ", old_weights[0], "\n\n",second_individual+" = ", old_weights[1], "\n")

        for j in range(len(new_weights)):
        
            for k in range(len(new_weights[j])):

                for i in range(len(new_weights[j][k])):
                    
                    random_var = random.randrange(0, 100, 1)

                    if random_var <= 50:
                        father = 0
                    
                    else:
                        father = 1
                    
                    new_weights[j][k][i] = old_weights[father][j][k][i]

        with open(third_individual, "wb") as file:
           pkl.dump((new_weights, number_layers), file) 
        #    print(new_weights, "\n\n")

        return None

    def clone(self, best):
        
        with open(best, "rb") as file:
            weights, number_layers = pkl.load(file)
        
        for i in range(len(self.name)):

            with open(self.name[i], "wb") as file:
                pkl.dump((weights, number_layers), file)
                
            self.mutation(self.name[i])
        
        return None

    def mutation(self, name):
        """ atualiza alguns poucos pesos da rede neural, gerados pela 'weights()' usando o 'learn_rate' como fator multiplicativo para isso.
            imput   => name
            output  => none
        """
        
        with open(name, "rb") as file:
            weights, layers = pkl.load(file)

        for i in range(len(weights)):
            
            for j in range(len(weights[i])):
            
                for k in range(len(weights[i][j])):
            
                    random_mutation_var = random.randrange(0, 100, 1)/100
                    random_var = random.randrange(-100,100,1)/100

                    if random_mutation_var <= self.mutation_rate:
                        
                        weights[i][j][k] = random_var

        # print(weights, "   mutation\n\n")

        with open(name, "wb") as file:
            pkl.dump((weights, layers), file)
        
        return None

    def activate_func(self, array, tipe_func = "step"):
        ''' Recebe "n" valores em um vetor n-dimensional submete cada um deles em uma função f(x) e devolve o resultado em um vetor n_dimensional
            tipe_func = ["step", "linear", "sigmoide", "tanh", "Relu"]
            imput   => array(x,y,z,...)
            output  => array(f(x),f(y),f(z),...)
        '''

        array = [float(i) for i in array] 
        resp = []

        if tipe_func == "step":
            for i in array:
                if i > 0:
                    resp.append(1)
                else:
                    resp.append(0)
        
        if tipe_func == "linear":
            x = 3
            for i in array:
                resp.append(i * x)    

        if tipe_func == "sigmoide":
            for i in array:
                resp.append(1/(1 + np.exp(-1 * i)))

        if tipe_func == "tanh":
            for i in array:
                resp.append(2/(1 + np.exp(-2 * i)) - 1)

        if tipe_func == "Relu": # tenho que atualizar, a função está errada
            for i in array:
                resp.append(i*x)
                    
        return resp


    def file_checker(self, n_individuals):

        individuals = []

        for i in range(n_individuals):
                
            file_name = "Weights/weights["+ str(i+1) +"]_bord[" + str(self.lines) + "][" + str(self.columns) +"].pkl"
            
            try:    
                with open(file_name, "rb") as file:
                        test = pkl.load(file)
                individuals.append(file_name)
                resp = True
            except:
                self.weights(file_name)
                resp = False
            
        return resp, individuals

    def sum(self, array):
        """Soma os valores de um array"""

        resp = 0
        
        for i in array:
            resp += float(i)
        
        return resp

    def layers(self, input, weights, layer, number_layers, next_number_layers):
        """Recebe os ,imputs ,pesos , camada onde está, a proxima camada e calcula os valores da proxima camada."""

        output = []
        
        for j in range(next_number_layers):#4
            value = []
            for i in range(number_layers):#2
                if i == 0:
                    value.append((self.bias * weights[layer][j][i]))
                value.append((input[i])*weights[layer][j][i+1]) #gerar o primeiro valor na seg camada
            # print(value)

            output.append(self.sum(value))

        return output

    def generation(self, array, n_best):
        """Recebe um array com as ai's e quantos seram os melhores , usando o 'DNA' dos melhores sera gerado novos individuos.
            input => array, n_best
            output => None
        """

        for i in range(len(array) - n_best):

            random_ai = np.random.randint(0, n_best)
            while True:
                random_other_ai = np.random.randint(0, n_best)
                
                if random_ai != random_other_ai:
                    break
            
            # print(random_ai,random_other_ai ,array[random_ai][0], array[random_other_ai][0], array[(n_best) + i][0], "\n\n")
            self.crossover(array[random_ai][0], array[random_other_ai][0], array[(n_best) + i][0])
            self.mutation(array[(n_best) + i][0])
        
        return None


    def run(self, input, name): #no futuro irei mudar isso pra algo automatico, o codigo esta sendo overfit no meu problema
        ''' Esse método recebe as entradas para a rede neural, ela usa os pesos gerados em "weights()" para calcular os resultados da camada interna.
            Com os resultados da camada interna usa a "activate_func()" para obter os valores e repete o processo para a ultiam camada obtendo a resposta da rede neural.
            imput   => bord_data
            output  => neural_network_response
        '''

        file_name = name
        
        with open(file_name, "rb") as file:
            weights, number_layers = pkl.load(file)
        
        for n in range(len(number_layers) - 1):
            here = number_layers[n]
            next = number_layers[n + 1]

            if n < len(number_layers) - 2:
                input = self.activate_func(self.layers(input, weights, n, here, next), "sigmoide")
            
            else:
                input = self.activate_func(self.layers(input, weights, n, here, next), "sigmoide")
            
        # print("imput = ",input)

        return(input)