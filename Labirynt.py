import numpy as np
import pygad as pg
import random, time

labirynth = np.array([
    ['#','#','#','#','#','#','#','#','#','#','#','#'],
    ['#','S','O','O','#','O','O','O','#','O','O','#'],
    ['#','#','#','O','O','O','#','O','#','#','0','#'],
    ['#','O','O','O','#','O','#','O','O','O','O','#'],
    ['#','O','#','O','#','#','O','O','#','#','O','#'],
    ['#','O','O','#','#','O','O','O','#','O','O','#'],
    ['#','O','O','O','O','O','#','O','O','O','#','#'],
    ['#','O','#','O','O','#','#','O','#','O','O','#'],
    ['#','O','#','#','#','O','O','O','#','#','O','#'],
    ['#','O','#','O','#','#','O','#','O','#','O','#'],
    ['#','O','#','O','O','O','O','O','O','O','E','#'],
    ['#','#','#','#','#','#','#','#','#','#','#','#']])

def logic_gate(small_error, big_error):
    if small_error and big_error:
        return False
    elif small_error and not big_error:
        return True
    elif not small_error and big_error:
        return False
    else:
        return True

'''gene_space = []
while len(gene_space) < 30:
    move = random.randint(1,4)
    if len(gene_space) == 0:
        gene_space.append(move)
    elif move == 1 and gene_space[-1] != 3:
        gene_space.append(move)
    elif move == 2 and gene_space[-1] != 4:
        gene_space.append(move)
    elif move == 3 and gene_space[-1] != 1:
        gene_space.append(move)
    elif move == 4 and gene_space[-1] != 2:
        gene_space.append(move)
    else:
        move = random.randint(1,4)'''

#geny: pokonana droga, maksymalnie 30 krokow, kazde wejscie w scianie -10 pkt
#gene_space: [koordynaty]
path_history = [[1,1]]
walls = 0
fups = 0

def fitness(ga_instance, solution, solution_index):
    position = [1,1]
    fitness_sum = 0
    for move in solution:
        if move == 1 and position[0] > 0 and path_history[-1][0] != position[0]-1:# and labirynth[position[0]-1, position[1]] != '#':#
            position[0] -= 1
            path_history.append(position)
        elif move == 2 and position[1] < 11 and path_history[-1][1] != position[1]+1:# and labirynth[position[0], position[1]+1] != '#':#
            position[1] += 1
            path_history.append(position)
        elif move == 3 and position[0] < 11 and path_history[-1][0] != position[0]+1:# and labirynth[position[0]+1, position[1]] != '#':#
            position[0] += 1
            path_history.append(position)
        elif move == 4 and position[1] > 0 and path_history[-1][1] != position[1]-1:# and labirynth[position[0], position[1]-1] != '#':#
            position[1] -= 1
            path_history.append(position)

        if labirynth[position[0], position[1]] == 'O':
            fitness_sum += 5
        elif labirynth[position[0], position[1]] == 'O' and position in path_history:
            fitness_sum -= 50
        elif labirynth[position[0], position[1]] == '#':
            fitness_sum -= 50
        elif labirynth[position[0], position[1]] == 'E':
            fitness_sum += 500
            return fitness_sum
            
    else:
        return fitness_sum
    
# 1 gora 2 prawo 3 dol 4 lewo
whole_time = 0
solution_fitness = 0
i = 0

for _ in range(10):
    start_time = time.time()
    ga_instance = pg.GA(
        num_generations=200,
        num_parents_mating=20,
        fitness_func=fitness,
        sol_per_pop=200,
        num_genes=30,
        gene_space=[1,2,3,4],
        crossover_type='two_points',
        mutation_probability=0.05,
        parent_selection_type='rank',
        #stop_criteria='saturate_100',
        keep_elitism=2
    )

    ga_instance.run()

    finish_time = int(time.time()-start_time)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution: {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
#print(f'Time: {finish_time} seconds')
#print(f'Walls: {walls} fuck ups: {fups}')
#whole_time += finish_time
#finish_time = 0
    #ga_instance.plot_fitness()

    #print(labirynth[10, 10])

    #print([1,2] in path_history)
'''    i +=1
    fups = 0
    walls = 0
else:
    print(f'It took mean: {whole_time/i+1} seconds')
    print(f'At all: {whole_time}')'''
