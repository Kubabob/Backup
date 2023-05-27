from ellipsometry_math import Elip_Structure
import pygad as pg
from numpy import *
import time
import matplotlib.pyplot as plt
import pandas as pd

def wave_length_spectrum(angle, thickness):
    Si = pd.read_csv('Si_nk.csv', sep=',')
    SiO2 = pd.read_csv('SiO2_nk.csv', sep=',')


    psis = {}
    deltas = {}
    for index, wave_length in enumerate(SiO2['wvl']):
        structure = Elip_Structure(angle, 0, thickness, (1,0), (float(SiO2['n'][index]), 0), (float(Si['n'][index]), float(Si['k'][index])))
        psis[wave_length] = structure.psi(r_p=structure.r_ijk_p(wave_length=wave_length), r_s=structure.r_ijk_s(wave_length=wave_length))   
        deltas[wave_length] = structure.delta(r_p=structure.r_ijk_p(wave_length=wave_length), r_s=structure.r_ijk_s(wave_length=wave_length))
    else:
        plt.plot(psis.keys(), psis.values())
        plt.plot(deltas.keys(), deltas.values())
        plt.grid()
        plt.show()
        print(f'Psi and delta for 0.6um: {psis[0.6]} {deltas[0.6]}\nPsi and delta for 0.7um: {psis[0.7]} {deltas[0.7]}')

def fitness_func(ga_solution, solution, solution_idx):
    model_psi_451 = 0.12931447242366356
    model_delta_451 = 3.617999055054228
    model_psi_551 = 0.7740248774457947
    model_delta_551 = 5.410573333883566

    #Si = pd.read_csv('Si_nk.csv')
    Si_n_451 = 4.6680
    Si_k_451 = 0.14650
    Si_n_551 = 4.0837
    Si_k_551 = 0.040229
    cauchy_plot = {}

    cauchy_plot[0.451] = solution[0] + solution[1]/0.451**2 + solution[2]/0.451**4
    cauchy_plot[0.551] = solution[0] + solution[1]/0.551**2 + solution[2]/0.551**4

    if cauchy_plot[0.451] != 0 and cauchy_plot[0.551] != 0:
        exp_structure1 = Elip_Structure(70, 0.451, solution[3], (1,0), (cauchy_plot[0.451], solution[4]), (Si_n_451, Si_k_451))
        exp_structure2 = Elip_Structure(70, 0.551, solution[3], (1,0), (cauchy_plot[0.551], solution[5]), (Si_n_551, Si_k_551))

        return 1-sqrt((exp_structure1.psi(r_p=exp_structure1.r_ijk_p(wave_length=0.451), r_s=exp_structure1.r_ijk_s(wave_length=0.451)) - model_psi_451)**2
                    + (exp_structure1.delta(r_p=exp_structure1.r_ijk_p(wave_length=0.451), r_s=exp_structure1.r_ijk_s(wave_length=0.451)) - model_delta_451)**2
                    + (exp_structure2.psi(r_p=exp_structure2.r_ijk_p(wave_length=0.551), r_s=exp_structure2.r_ijk_s(wave_length=0.551)) - model_psi_551)**2
                    + (exp_structure2.delta(r_p=exp_structure2.r_ijk_p(wave_length=0.551), r_s=exp_structure2.r_ijk_s(wave_length=0.551)) - model_delta_551)**2)*10
    else:
        return -10



def GA_1():
    ellipsometry_get_thickness = pg.GA(
        num_generations=30,
        num_parents_mating=100,
        fitness_func=fitness_func,
        sol_per_pop=1000,
        num_genes=6,
        #A B C thickness k1 k2
        gene_space=[arange(1,5,0.005), arange(0.001,1,0.0005), arange(0,1,0.01), arange(0,1,0.0005), arange(0, 5, 0.01), arange(0, 5, 0.01)],
        parent_selection_type='rank',
        keep_elitism=10,
        mutation_type='random',
        mutation_probability=0.5,
        random_mutation_min_val=-0.03,
        random_mutation_max_val=0.03,
    )

    start_time = time.time()
    ellipsometry_get_thickness.run()

    solution, solution_fitness, solution_idx = ellipsometry_get_thickness.best_solution()
    print(f"1\nParameters of the best solution: {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f'It took: {time.time() - start_time} seconds\n')

    return solution

def GA_2(solution):
    ellipsometry_get_thickness2 = pg.GA(
        num_generations=50,
        num_parents_mating=100,
        fitness_func=fitness_func,
        sol_per_pop=1000,
        num_genes=6,
        gene_space=[arange(solution[0]-0.1,solution[0]+0.1,0.001), arange(solution[1]-0.01,solution[1]+0.01,0.0001), arange(solution[2]-0.001, solution[2]+0.001, 0.001), arange(solution[3]-0.01,solution[3]+0.01,0.00001), arange(solution[4]-0.01, solution[4]+0.01, 0.001), arange(solution[5]-0.01, solution[5]+0.01, 0.001)],
        parent_selection_type='rank',
        keep_elitism=10,
        mutation_type='random',
        mutation_probability=0.3,
        random_mutation_min_val=-0.03,
        random_mutation_max_val=0.03,
    )

    start_time = time.time()
    ellipsometry_get_thickness2.run()

    solution2, solution_fitness, solution_idx = ellipsometry_get_thickness2.best_solution()
    print(f"2\nParameters of the best solution: {solution2}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f'It took: {time.time() - start_time} seconds\n')

def get_thickness():
    for _ in range(10):
        GA_2(GA_1())

get_thickness()