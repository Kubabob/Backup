from vendor import Vendor
from population import Population
import time, os, sys
'''
the results gotten from testing population are the best 25%
now we need to combine genes of 1st, 2nd; 3rd, 4th etc. in following order:
    child 1. Gene(A1), gene(A2), Gene(B3), gene(B4)
    child 2. gene(B1), gene(B2), gene(A3), gene(A4)
    child 3. full A
    child 4. full B

    last two are check whether combinations are positive
'''

class Civilization:
    def __init__(self, tests_for_each_population: int, steps_to_reach: int, number_of_populations: int) -> None:
        self.tests_for_each_population = tests_for_each_population
        self.number_of_populations = number_of_populations
        self.steps_to_reach = steps_to_reach
        self.last_population_results: list
        pass

    def simulation(self):
        simulation_start_time = int(time.time())
        for i in range(0, self.number_of_populations, 1):
            population = Population(i)
            if i == 0:
                population.create_population()
            else:
                population.recombine(self.last_population_results)        
            self.show_actual_parameters(i, simulation_start_time)
            population.test_population(test_cities=[(5,0), (5,5), (10,5), (10,10)], how_many_times=self.tests_for_each_population, how_many_bests=20)
            self.last_population_results = population.best_of_population
            
        else:
            print(f'The best combination of {i+1} populations is: {self.last_population_results[0]} with steps: {list(population.vendor_ranking.values())[0]}')

    def show_actual_parameters(self, population_number, start_time):
        clear = lambda: os.system('cls')
        clear()
        sys.stdout.write(f'Population number: {population_number+1}\n')
        #sys.stdout.write(f'Time since started: {int(time.time() - start_time)//60} minutes {int(time.time() - start_time)%60} seconds\n')
        sys.stdout.flush()
            
                
def main():
    civilization = Civilization(tests_for_each_population=2, steps_to_reach=70, number_of_populations=10)
    civilization.simulation()

if __name__ == '__main__':
    main()
