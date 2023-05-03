from ellipsometry_math import Ellipsometry
import pygad as pg
from numpy import *
import time
import matplotlib.pyplot as plt

def get_thickness_fitness(ga_instance, solution, solution_index):
    angle = 70
    wave_length = 0.6328
    N0 = (1.00027653, 0)
    #N1 = (1.4851, 0)
    #N2 = (5.0908, 3.6265)

    N1 = (1.4570, 0)
    N2 = (3.8827, 0.019626)

    ellipsometry_model = Ellipsometry(angle, wave_length, None, N0, N1, N2)

    ellipsometry_exp = Ellipsometry(angle, wave_length, float(solution), N0, N1, N2)

    '''model_r_ijk_p = ellipsometry_model.r_ijk_p(beta=ellipsometry_model.beta(Ni=ellipsometry_model.complex_refractive_indexes[1], Nj=ellipsometry_model.complex_refractive_indexes[2]))
    model_r_ijk_s = ellipsometry_model.r_ijk_s(beta=ellipsometry_model.beta(Ni=ellipsometry_model.complex_refractive_indexes[1], Nj=ellipsometry_model.complex_refractive_indexes[2]))
    model_psi = ellipsometry_model.psi(
        r_p=model_r_ijk_p,
        r_s=model_r_ijk_s)
    model_delta = ellipsometry_model.delta(
        r_p=model_r_ijk_p,
        r_s=model_r_ijk_s,
        psi=model_psi
    )'''

    model_psi = 38
    model_delta = 20

    exp_beta = ellipsometry_exp.beta(i=0, j=1)
    
    exp_r_ijk_p = ellipsometry_exp.r_ijk_p(i=0,j=1,k=2)
    exp_r_ijk_s = ellipsometry_exp.r_ijk_s(i=0,j=1,k=2)
    exp_psi = ellipsometry_exp.psi(
            r_p=exp_r_ijk_p,
            r_s=exp_r_ijk_p
            )
    exp_delta = ellipsometry_exp.delta(
        r_p=exp_r_ijk_p,
        r_s=exp_r_ijk_s
        )

    return -(abs(model_psi - exp_psi) + abs(model_delta - exp_delta))

ellipsometry_get_thickness = pg.GA(
    num_generations=200,
    num_parents_mating=20,
    fitness_func=get_thickness_fitness,
    sol_per_pop=200,
    num_genes=1,
    gene_space=[list(arange(0, 1000, 1))],
    parent_selection_type='rank',
    keep_elitism=4,
    mutation_type='random',
    mutation_probability=0.1,
)

'''start_time = time.time()
ellipsometry_get_thickness.run()

solution, solution_fitness, solution_idx = ellipsometry_get_thickness.best_solution()
print(f"Parameters of the best solution: {solution}")
print(f"Fitness value of the best solution = {solution_fitness}\n")
print(f'It took: {time.time() - start_time} seconds')

ellipsometry_get_thickness.plot_fitness()'''

N0 = (1.00027653, 0)
#N1 = (1.4851, 0)
#N2 = (5.0908, 3.6265)

N1 = (1.4570, 0)
N2 = (3.8827, 0.019626)

'''ellipsometry_exp = Ellipsometry(70, 0.6328, None, N0, N1, N2)

exp_beta = ellipsometry_exp.beta(Ni=ellipsometry_exp.complex_refractive_indexes[1], Nj=ellipsometry_exp.complex_refractive_indexes[2])


exp_r_ijk_p = ellipsometry_exp.r_ijk_p(beta=exp_beta)
exp_r_ijk_s = ellipsometry_exp.r_ijk_s(beta=exp_beta)
exp_psi = ellipsometry_exp.psi(
        r_p=exp_r_ijk_p,
        r_s=exp_r_ijk_p)
exp_delta = ellipsometry_exp.delta(
    r_p=exp_r_ijk_p,
    r_s=exp_r_ijk_s,
    psi=exp_psi
)'''

psi = []
delta = []
x = [i for i in range(100)]

for i in range(100):
    ellipsometry_exp = Ellipsometry(70, 0.6328, i, N0, N1, N2)
    #exp_beta = ellipsometry_exp.beta(i=0,j=1)
    exp_r_ijk_p = ellipsometry_exp.r_ij_p()
    exp_r_ijk_s = ellipsometry_exp.r_ij_s()
    exp_psi = ellipsometry_exp.psi(
        r_p=exp_r_ijk_p,
        r_s=exp_r_ijk_s)
    exp_delta = ellipsometry_exp.delta(
        r_p=exp_r_ijk_p,
        r_s=exp_r_ijk_s,
    )

    psi.append(exp_psi)
    delta.append(exp_delta)
else:
    plt.plot(psi)
    plt.plot(delta)
    plt.show()
