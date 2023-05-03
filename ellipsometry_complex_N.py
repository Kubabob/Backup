from ellipsometry_math import Ellipsometry
import pygad as pg
from numpy import *
import time

def get_complex_N_fitness(ga_instance, solution, solution_index):
    angle = 70
    wave_length = 0.6328
    N0 = (1.00027653, 0)
    N1_exp = (float(solution[0]), float(solution[1]))
    N1_model = (3.8827, 0.019626)
    ellipsometry_model = Ellipsometry(angle, wave_length, N0, N1_model)

    ellipsometry_exp = Ellipsometry(angle, wave_length, N0, N1_exp)

    #print(solution)

    return -abs(ellipsometry_model.tan_psi_exp_mdeltai() - ellipsometry_exp.tan_psi_exp_mdeltai())

ellipsometry_get_complex_N = pg.GA(
    num_generations=100,
    num_parents_mating=20,
    fitness_func=get_complex_N_fitness,
    sol_per_pop=200,
    num_genes=2,
    gene_space=[list(arange(1, 8, 0.001)), list(arange(0, 6, 0.001))],
    parent_selection_type='rank',
    keep_elitism=4,
    mutation_type='random',
    mutation_probability=0.1,
    stop_criteria='saturate_20'
)
start_time = time.time()
ellipsometry_get_complex_N.run()

solution, solution_fitness, solution_idx = ellipsometry_get_complex_N.best_solution()
print(f"Parameters of the best solution: {solution}")
print(f"Fitness value of the best solution = {solution_fitness}\n")
print(f'It took: {time.time() - start_time} seconds')

ellipsometry_get_complex_N.plot_fitness()

