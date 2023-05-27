from ellipsometry_math import Elip_Structure
import pygad as pg
from numpy import *
import time
import matplotlib.pyplot as plt
import pandas as pd

'''def get_thickness_fitness(ga_instance, solution, solution_index):
    angle = 75
    angle2 = 65
    wave_length = 0.635
    N0 = (1, 0)
    N1 = (1.4570, 0)
    N2 = (3.8787, 0.019221)


    ellipsometry_exp = Elip_Structure(angle, wave_length, float(solution), N0, N1, N2)
    ellipsometry_exp2 = Elip_Structure(angle2, wave_length, float(solution), N0, N1, N2)


    model_psi = 128.69612
    model_delta = 59.90756

    model_psi2 = 43.52267
    model_delta2 = 41.20184

    
    exp_r_ijk_p = ellipsometry_exp.r_ijk_p(i=0,j=1,k=2)
    exp_r_ijk_s = ellipsometry_exp.r_ijk_s(i=0,j=1,k=2)
    exp_psi = rad2deg(ellipsometry_exp.psi(
            r_p=exp_r_ijk_p,
            r_s=exp_r_ijk_s
            ))
    exp_delta = rad2deg(ellipsometry_exp.delta(
        r_p=exp_r_ijk_p,
        r_s=exp_r_ijk_s
        ))
    
    exp_r_ijk_p2 = ellipsometry_exp2.r_ijk_p(i=0,j=1,k=2)
    exp_r_ijk_s2 = ellipsometry_exp2.r_ijk_s(i=0,j=1,k=2)
    exp_psi2 = rad2deg(ellipsometry_exp2.psi(
            r_p=exp_r_ijk_p2,
            r_s=exp_r_ijk_s2
            ))
    exp_delta2 = rad2deg(ellipsometry_exp2.delta(
        r_p=exp_r_ijk_p2,
        r_s=exp_r_ijk_s2
        ))

    #return -abs(ellipsometry_exp.tan_psi_exp_mdeltai(psi=model_psi, delta=model_delta) - ellipsometry_exp.tan_psi_exp_mdeltai(r_p=exp_r_ijk_p, r_s=exp_r_ijk_s))
    return -(abs(model_psi - exp_psi) + abs(model_delta - exp_delta) + abs(model_psi2 - exp_psi2) + abs(model_delta2 - exp_delta2))

ellipsometry_get_thickness = pg.GA(
    num_generations=100,
    num_parents_mating=20,
    fitness_func=get_thickness_fitness,
    sol_per_pop=200,
    num_genes=1,
    gene_space=[list(arange(0, 1, 0.01))],
    parent_selection_type='rank',
    keep_elitism=4,
    mutation_type='random',
    mutation_probability=0.1,
)'''

'''start_time = time.time()
ellipsometry_get_thickness.run()

solution, solution_fitness, solution_idx = ellipsometry_get_thickness.best_solution()
print(f"Parameters of the best solution: {solution}")
print(f"Fitness value of the best solution = {solution_fitness}\n")
print(f'It took: {time.time() - start_time} seconds')

ellipsometry_get_thickness.plot_fitness()'''

N0 = (1, 0)
N1 = (1.4570, 0)
N2 = (3.8787, 0.019221)


#plot of psi and delta diffs with thickness change
'''psi = []
delta = []
x = [i for i in range(0, 1000, 1)]

for thickness_nm in range(0, 1000, 1):
    thickness_um = thickness_nm / 1000
    ellipsometry_exp = Ellipsometry(75, 0.635, thickness_um, N0, N1, N2)
    #exp_beta = ellipsometry_exp.beta(i=0,j=1)
    exp_r_ijk_p = ellipsometry_exp.r_ijk_p()
    exp_r_ijk_s = ellipsometry_exp.r_ijk_s()
    exp_psi = ellipsometry_exp.psi(
        r_p=exp_r_ijk_p,
        r_s=exp_r_ijk_s)
    exp_delta = ellipsometry_exp.delta(
        r_p=exp_r_ijk_p,
        r_s=exp_r_ijk_s,
    )

    psi.append(rad2deg(exp_psi))
    delta.append(rad2deg(exp_delta))
else:
    plt.plot(x, psi)
    plt.plot(x, delta)
    plt.grid()
    plt.show()'''

'''
#plot of psi and delta diffs of wave_length change
psi = []
delta = []
x = [i for i in range(250, 1250, 1)]

for wave_length_nm in range(250, 1250, 1):
    wave_length_um = wave_length_nm / 1000
    ellipsometry_exp = Ellipsometry(75, wave_length_um, 0.1, N0, N1, N2)
    #exp_beta = ellipsometry_exp.beta(i=0,j=1)
    exp_r_ijk_p = ellipsometry_exp.r_ijk_p()
    exp_r_ijk_s = ellipsometry_exp.r_ijk_s()
    exp_psi = ellipsometry_exp.psi(
        r_p=exp_r_ijk_p,
        r_s=exp_r_ijk_s)
    exp_delta = ellipsometry_exp.delta(
        r_p=exp_r_ijk_p,
        r_s=exp_r_ijk_s,
    )

    psi.append(rad2deg(exp_psi))
    delta.append(rad2deg(exp_delta))
else:
    plt.plot(x, psi)
    plt.plot(x, delta)
    plt.grid()
    plt.show()'''

#print(rad2deg(0.6132))

#plot of fitness change with thickness change
#results = []
def get_thickness(angle: int = 0):
    fitness_results = {}
    psis = {}
    deltas = {}
    x = arange(0, 1000, 1)
    for thickness_nm in x:
        #for the sake of math formulas i change nm to um
        thickness_um = thickness_nm/1000

        #4 angle options in degrees
        if angle == 0:
            angle1 = 45

            #600 45 deg
            model_psi = 34.66707
            model_delta = 18.1176
            #700 45 deg
            model_psi2 = 34.1864
            model_delta2 = 17.05903
        elif angle == 1:
            angle1 = 55

            #600 55 deg
            model_psi = 180.0273
            model_delta = 179.7674
            #700 55 deg
            model_psi2 = 179.9559
            model_delta2 = 179.95126
        elif angle == 2:
            angle1 = 65

            #600 65 deg
            model_psi = 28.0328
            model_delta = 1.67467
            #700 65 deg
            model_psi2 = 27.43605
            model_delta2 = 0.396
        elif angle == 3:
            angle1 = 75

            #600 75 deg
            model_psi = 179.80653
            model_delta = 177.64632
            #700 75 deg
            model_psi2 = 179.71513
            model_delta2 = 164.25052

        wave_length1 = 0.6
        wave_length2 = 0.7
        #complex refractive index of air
        N0 = (1, 0)
        
        #cri of SiO2 and Si 0.6 um
        N1 = (1.4542, 0)
        N2 = (3.7348, 0.0090921)

        #cri of SiO2 and Si 0.7 um
        N3 = (1.4553, 0)
        N4 = (3.7838, 0.012170)

        
        #we make 2 structures of 3 layers: air SiO2 Si
        exp_structure1 = Elip_Structure(angle1, wave_length1, thickness_um, N0, N1, N2)
        exp_structure2 = Elip_Structure(angle1, wave_length2, thickness_um, N0, N3, N4)
        

        #get p- and s- reflectances of first 3 layer structure
        exp_r_ijk_p = exp_structure1.r_ijk_p(i=0,j=1,k=2)
        exp_r_ijk_s = exp_structure1.r_ijk_s(i=0,j=1,k=2)
        #calculate psi and delta(in degrees) out of reflectances 
        exp_psi = rad2deg(exp_structure1.psi(
                r_p=exp_r_ijk_p,
                r_s=exp_r_ijk_s
                ))
        exp_delta = rad2deg(exp_structure1.delta(
            r_p=exp_r_ijk_p,
            r_s=exp_r_ijk_s
            ))

        #same for the 2nd structure
        exp_r_ijk_p2 = exp_structure2.r_ijk_p(i=0,j=1,k=2)
        exp_r_ijk_s2 = exp_structure2.r_ijk_s(i=0,j=1,k=2)
        exp_psi2 = rad2deg(exp_structure2.psi(
                r_p=exp_r_ijk_p2,
                r_s=exp_r_ijk_s2
                ))
        exp_delta2 = rad2deg(exp_structure2.delta(
            r_p=exp_r_ijk_p2,
            r_s=exp_r_ijk_s2
            ))
        
        #gather results
        fitness_results[thickness_nm] = -sqrt((((model_psi - exp_psi))**2 + ((model_delta - exp_delta))**2 + ((model_psi2 - exp_psi2))**2 + ((model_delta2 - exp_delta2))**2))
        psis[thickness_nm] = (exp_psi, exp_psi2)
        deltas[thickness_nm] = (exp_delta, exp_delta2)

    else:
        max_fitness = max(fitness_results.values())
        thickness = [i for i in fitness_results if fitness_results[i]==max_fitness]
        thickness = thickness[0]

        print(f"Thickness: {thickness} nm")
        print(f'Psis: {psis[thickness]}')
        print(f'Deltas: {deltas[thickness]}')
        
        plt.plot(x, fitness_results.values())
        plt.show()

def show_reflectance_plot():
    #750 nm 
    N0 = (1, 0)
    N1 = (1.4542, 0)
    N2 = (3.7348, 0.0090921)

    el = Elip_Structure(0, 0.75, 0.1, N0, N1, N2)
    el.reflectance_plot(3)

#get_thickness(angle=3)
#show_reflectance_plot()

'''el1 = Elip_Structure(0, 0.6199, 0, (1,0), (0.27, 3.2))
el2 = Elip_Structure(0, 0.6199, 0, (1,0), (3.906, 0.022))

el2.psi_delta_plot()'''

el3 = Elip_Structure(70, 0.6328, 0, (1,0), (1.46, 0), (3.87, 0.0146))
#el3.psi_delta_plot(layers=3, is_thickness=True)
#el3.reflectance_plot(layers=3)

#el4 = Elip_Structure(70, 0.1, 0.2, (1, 0), (1.41, 0), (2, 0))
#el4.psi_delta_plot(layers=3, is_wave_length=True)
def wave_length_spectrum(angle, thickness):
    Si = pd.read_csv('Si_nk.csv', sep=',')
    SiO2 = pd.read_csv('SiO2_nk.csv', sep=',')

    #print(SiO2['n'][0])
    #print(Si['k'][150])

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

#wave_length_spectrum(70, 0.45)

def fitness_func(ga_solution, solution, solution_idx):
    model_psi_600 = 0.7822723560360596
    model_delta_600 = 1.4043460136228663
    model_psi_700 = 0.6336531933817289
    model_delta_700 = 1.3751008061650751

    Si = pd.read_csv('Si_nk.csv')
    #wave_lengths = arange(0.25, 1, 0.005)
    cauchy_plot = {}
    '''for wave_length in wave_lengths:
        wave_length = round(wave_length, 3)
        cauchy_plot[wave_length] = solution[0] + solution[1]/wave_length**2 + solution[2]/wave_length**4
    '''
    cauchy_plot[0.6] = solution[0] + solution[1]/0.6**2 + solution[2]/0.6**4
    cauchy_plot[0.7] = solution[0] + solution[1]/0.7**2 + solution[2]/0.7**4

    #print(f'600: {cauchy_plot[0.6]}')
    #print(f'700: {cauchy_plot[0.7]}')
    if cauchy_plot[0.6] != 0 and cauchy_plot[0.7] != 0:
        exp_structure1 = Elip_Structure(70, 0.6, solution[3], (1,0), (cauchy_plot[0.6], 0), (Si['n'][70], Si['k'][70]))
        exp_structure2 = Elip_Structure(70, 0.7, solution[3], (1,0), (cauchy_plot[0.7], 0), (Si['n'][90], Si['k'][90]))

        return 1-sqrt((exp_structure1.psi(r_p=exp_structure1.r_ijk_p(wave_length=0.6), r_s=exp_structure1.r_ijk_s(wave_length=0.6)) - model_psi_600)**2
                    + (exp_structure1.delta(r_p=exp_structure1.r_ijk_p(wave_length=0.6), r_s=exp_structure1.r_ijk_s(wave_length=0.6)) - model_delta_600)**2
                    + (exp_structure2.psi(r_p=exp_structure2.r_ijk_p(wave_length=0.7), r_s=exp_structure2.r_ijk_s(wave_length=0.7)) - model_psi_700)**2
                    + (exp_structure2.delta(r_p=exp_structure2.r_ijk_p(wave_length=0.7), r_s=exp_structure2.r_ijk_s(wave_length=0.7)) - model_delta_700)**2)*10
    else:
        return -10




#for _ in range(10):

def GA_1():
    ellipsometry_get_thickness = pg.GA(
        num_generations=20,
        num_parents_mating=100,
        fitness_func=fitness_func,
        sol_per_pop=1000,
        num_genes=4,
        gene_space=[arange(1,2,0.005), arange(0.001,0.01,0.0005), arange(0,0.1,0.01), arange(0,1,0.0005)],
        parent_selection_type='rank',
        keep_elitism=5,
        mutation_type='random',
        mutation_probability=0.5,
        random_mutation_min_val=-0.03,
        random_mutation_max_val=0.03,
        #crossover_type='two_points',
        #random_seed=random.randint(1,10),
        #stop_criteria='reach_0.95'
    )

    start_time = time.time()
    ellipsometry_get_thickness.run()

    solution, solution_fitness, solution_idx = ellipsometry_get_thickness.best_solution()
    print(f"1\nParameters of the best solution: {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f'It took: {time.time() - start_time} seconds\n')

    return solution

#ellipsometry_get_thickness.plot_fitness()
def GA_2(solution):
    ellipsometry_get_thickness2 = pg.GA(
        num_generations=20,
        num_parents_mating=100,
        fitness_func=fitness_func,
        sol_per_pop=1000,
        num_genes=4,
        gene_space=[arange(solution[0]-0.1,solution[0]+0.1,0.001), arange(solution[1]-0.01,solution[1]+0.01,0.0001), arange(solution[2]-0.001, solution[2]+0.001, 0.001), arange(solution[3]-0.01,solution[3]+0.01,0.00001)],
        parent_selection_type='rank',
        keep_elitism=5,
        mutation_type='random',
        mutation_probability=0.3,
        random_mutation_min_val=-0.03,
        random_mutation_max_val=0.03,
        #crossover_type='two_points',
        #random_seed=random.randint(1,10),
        #stop_criteria='reach_0.95'
    )

    start_time = time.time()
    ellipsometry_get_thickness2.run()

    solution2, solution_fitness, solution_idx = ellipsometry_get_thickness2.best_solution()
    print(f"2\nParameters of the best solution: {solution2}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f'It took: {time.time() - start_time} seconds\n')

    #ellipsometry_get_thickness2.plot_fitness()

'''for _ in range(10):
    GA_2(GA_1())'''

def grid_search(solution):
    model_psi_600 = 0.6340659294825799
    model_delta_600 = 4.881929806502101
    model_psi_700 = 1.0735215481969287
    model_delta_700 = 1.4473917434606296

    Si = pd.read_csv('Si_nk.csv')
    #wave_lengths = arange(0.25, 1, 0.005)
    cauchy_plot = {}
    '''for wave_length in wave_lengths:
        wave_length = round(wave_length, 3)
        cauchy_plot[wave_length] = solution[0] + solution[1]/wave_length**2 + solution[2]/wave_length**4
    '''
    cauchy_plot[0.6] = solution[0] + solution[1]/0.6**2 + solution[2]/0.6**4
    cauchy_plot[0.7] = solution[0] + solution[1]/0.7**2 + solution[2]/0.7**4

    fitness_results = {}

    thickness_range = arange(solution[3]-0.02, solution[3]+0.02, 0.0001)
    best_fitness = -1

    for idx, thickness in enumerate(thickness_range):
        if cauchy_plot[0.6] != 0 and cauchy_plot[0.7] != 0:
            exp_structure1 = Elip_Structure(70, 0.6, thickness, (1,0), (cauchy_plot[0.6], 0), (Si['n'][70], Si['k'][70]))
            exp_structure2 = Elip_Structure(70, 0.7, thickness, (1,0), (cauchy_plot[0.7], 0), (Si['n'][90], Si['k'][90]))

            fitness_results[thickness] = 1-sqrt((exp_structure1.psi(r_p=exp_structure1.r_ijk_p(wave_length=0.6), r_s=exp_structure1.r_ijk_s(wave_length=0.6)) - model_psi_600)**2
                        + (exp_structure1.delta(r_p=exp_structure1.r_ijk_p(wave_length=0.6), r_s=exp_structure1.r_ijk_s(wave_length=0.6)) - model_delta_600)**2
                        + (exp_structure2.psi(r_p=exp_structure2.r_ijk_p(wave_length=0.7), r_s=exp_structure2.r_ijk_s(wave_length=0.7)) - model_psi_700)**2
                        + (exp_structure2.delta(r_p=exp_structure2.r_ijk_p(wave_length=0.7), r_s=exp_structure2.r_ijk_s(wave_length=0.7)) - model_delta_700)**2) 
            
            '''if new_fitness > best_fitness:
                best_fitness = new_fitness
            else:
                return thickness_range[idx-1]'''
    else:
        max_fitness = max(fitness_results.values())
        thickness = [i for i in fitness_results if fitness_results[i]==max_fitness]
        thickness = thickness[0]
        plt.plot(fitness_results.keys(), fitness_results.values())
        plt.grid()
        plt.show()
        return thickness
        

#print(grid_search([1.45, 0.006, 0, 0.449]))
    


#Si = pd.read_csv('Si_nk.csv')
#print(Si['n'][70])
#print(Si['n'][90])

'''wave_lengths = arange(0.25, 1, 0.005)
cauchy_plot = {}
for wave_length in wave_lengths:
    wave_length = round(wave_length, 3)
    cauchy_plot[wave_length] = 2 + 2/wave_length**2 + 2/wave_length**4

print(f'600: {cauchy_plot[0.6]}')
print(f'700: {cauchy_plot[0.7]}')'''

def gradient_search(solution):
    model_psi_600 = 0.6340659294825799
    model_delta_600 = 4.881929806502101
    model_psi_700 = 1.0735215481969287
    model_delta_700 = 1.4473917434606296
    Si = pd.read_csv('Si_nk.csv')

    

    for idx, x in enumerate(solution):
        if idx == 0:
            results_0 = {}
            for y in arange(x-0.2, x+0.2, 0.001):
                y = round(y, 4)

                #wave_lengths = arange(0.25, 1, 0.005)
                cauchy_plot = {}
                '''for wave_length in wave_lengths:
                    wave_length = round(wave_length, 3)
                    cauchy_plot[wave_length] = solution[0] + solution[1]/wave_length**2 + solution[2]/wave_length**4
                '''
                cauchy_plot[0.6] = y + solution[1]/0.6**2 + solution[2]/0.6**4
                cauchy_plot[0.7] = y + solution[1]/0.7**2 + solution[2]/0.7**4

                #print(f'600: {cauchy_plot[0.6]}')
                #print(f'700: {cauchy_plot[0.7]}')
                if cauchy_plot[0.6] != 0 and cauchy_plot[0.7] != 0:
                    exp_structure1 = Elip_Structure(70, 0.6, solution[3], (1,0), (cauchy_plot[0.6], 0), (Si['n'][70], Si['k'][70]))
                    exp_structure2 = Elip_Structure(70, 0.7, solution[3], (1,0), (cauchy_plot[0.7], 0), (Si['n'][90], Si['k'][90]))

                    results_0[y] = 1-sqrt((exp_structure1.psi(r_p=exp_structure1.r_ijk_p(wave_length=0.6), r_s=exp_structure1.r_ijk_s(wave_length=0.6)) - model_psi_600)**2
                                + (exp_structure1.delta(r_p=exp_structure1.r_ijk_p(wave_length=0.6), r_s=exp_structure1.r_ijk_s(wave_length=0.6)) - model_delta_600)**2
                                + (exp_structure2.psi(r_p=exp_structure2.r_ijk_p(wave_length=0.7), r_s=exp_structure2.r_ijk_s(wave_length=0.7)) - model_psi_700)**2
                                + (exp_structure2.delta(r_p=exp_structure2.r_ijk_p(wave_length=0.7), r_s=exp_structure2.r_ijk_s(wave_length=0.7)) - model_delta_700)**2)*10
                else:
                    results_0[y] = -10
        
        elif idx == 1:
            results_1 = {}
            for y in arange(x-0.2, x+0.2, 0.0001):
                y = round(y, 4)

                #wave_lengths = arange(0.25, 1, 0.005)
                cauchy_plot = {}
                '''for wave_length in wave_lengths:
                    wave_length = round(wave_length, 3)
                    cauchy_plot[wave_length] = solution[0] + solution[1]/wave_length**2 + solution[2]/wave_length**4
                '''
                cauchy_plot[0.6] = solution[0] + y/0.6**2 + solution[2]/0.6**4
                cauchy_plot[0.7] = solution[0] + y/0.7**2 + solution[2]/0.7**4

                #print(f'600: {cauchy_plot[0.6]}')
                #print(f'700: {cauchy_plot[0.7]}')
                if cauchy_plot[0.6] != 0 and cauchy_plot[0.7] != 0:
                    exp_structure1 = Elip_Structure(70, 0.6, solution[3], (1,0), (cauchy_plot[0.6], 0), (Si['n'][70], Si['k'][70]))
                    exp_structure2 = Elip_Structure(70, 0.7, solution[3], (1,0), (cauchy_plot[0.7], 0), (Si['n'][90], Si['k'][90]))

                    results_1[y] = 1-sqrt((exp_structure1.psi(r_p=exp_structure1.r_ijk_p(wave_length=0.6), r_s=exp_structure1.r_ijk_s(wave_length=0.6)) - model_psi_600)**2
                                + (exp_structure1.delta(r_p=exp_structure1.r_ijk_p(wave_length=0.6), r_s=exp_structure1.r_ijk_s(wave_length=0.6)) - model_delta_600)**2
                                + (exp_structure2.psi(r_p=exp_structure2.r_ijk_p(wave_length=0.7), r_s=exp_structure2.r_ijk_s(wave_length=0.7)) - model_psi_700)**2
                                + (exp_structure2.delta(r_p=exp_structure2.r_ijk_p(wave_length=0.7), r_s=exp_structure2.r_ijk_s(wave_length=0.7)) - model_delta_700)**2)*10
                else:
                    results_1[y] = -10
            
        elif idx == 2:
            pass

        elif idx == 3:
            results_3 = {}
            for y in arange(x-0.2, x+0.2, 0.00001):
                y = round(y, 4)

                #wave_lengths = arange(0.25, 1, 0.005)
                cauchy_plot = {}
                '''for wave_length in wave_lengths:
                    wave_length = round(wave_length, 3)
                    cauchy_plot[wave_length] = solution[0] + solution[1]/wave_length**2 + solution[2]/wave_length**4
                '''
                cauchy_plot[0.6] = solution[0] + solution[1]/0.6**2 + solution[2]/0.6**4
                cauchy_plot[0.7] = solution[0] + solution[1]/0.7**2 + solution[2]/0.7**4

                #print(f'600: {cauchy_plot[0.6]}')
                #print(f'700: {cauchy_plot[0.7]}')
                if cauchy_plot[0.6] != 0 and cauchy_plot[0.7] != 0:
                    exp_structure1 = Elip_Structure(70, 0.6, y, (1,0), (cauchy_plot[0.6], 0), (Si['n'][70], Si['k'][70]))
                    exp_structure2 = Elip_Structure(70, 0.7, y, (1,0), (cauchy_plot[0.7], 0), (Si['n'][90], Si['k'][90]))

                    results_3[y] = 1-sqrt((exp_structure1.psi(r_p=exp_structure1.r_ijk_p(wave_length=0.6), r_s=exp_structure1.r_ijk_s(wave_length=0.6)) - model_psi_600)**2
                                + (exp_structure1.delta(r_p=exp_structure1.r_ijk_p(wave_length=0.6), r_s=exp_structure1.r_ijk_s(wave_length=0.6)) - model_delta_600)**2
                                + (exp_structure2.psi(r_p=exp_structure2.r_ijk_p(wave_length=0.7), r_s=exp_structure2.r_ijk_s(wave_length=0.7)) - model_psi_700)**2
                                + (exp_structure2.delta(r_p=exp_structure2.r_ijk_p(wave_length=0.7), r_s=exp_structure2.r_ijk_s(wave_length=0.7)) - model_delta_700)**2)*10
                else:
                    results_3[y] = -10

    else:
        print(f'A: {max(results_0, key=results_0.get)}')
        print(f'B: {max(results_1, key=results_1.get)}')
        print(f'C: 0')
        print(f'Thickness: {max(results_3, key=results_3.get)}')

#gradient_search(solution)

el7 = Elip_Structure(70, 0.451, 0.3, (1,0), (1.75, 0.041), (4.6680, 0.14650))
el8 = Elip_Structure(70, 0.551, 0.3, (1,0), (1.74, 0.012), (4.0837, 0.040229))

model_psi_451nm_300nm = el7.psi(r_p=el7.r_ijk_p(wave_length=0.451, thickness=0.3), r_s=el7.r_ijk_s(wave_length=0.451, thickness=0.3))
model_delta_451nm_300nm = el7.delta(r_p=el7.r_ijk_p(wave_length=0.451, thickness=0.3), r_s=el7.r_ijk_s(wave_length=0.451, thickness=0.3))

model_psi_551nm_300nm = el8.psi(r_p=el8.r_ijk_p(wave_length=0.551, thickness=0.3), r_s=el8.r_ijk_s(wave_length=0.551, thickness=0.3))
model_delta_551nm_300nm = el8.delta(r_p=el8.r_ijk_p(wave_length=0.551, thickness=0.3), r_s=el8.r_ijk_s(wave_length=0.551, thickness=0.3))


'''print(f'model psi 451nm 300nm: {model_psi_451nm_300nm}')
print(f'model delta 451nm 300nm: {model_delta_451nm_300nm}\n')
print(f'model psi 551nm 300nm: {model_psi_551nm_300nm}')
print(f'model delta 551nm 300nm: {model_delta_551nm_300nm}')'''

el7.psi_delta_plot(layers=3, is_thickness=True)
