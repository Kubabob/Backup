import random
import os, sys, time

'''
Creating class of vendor that travels between 4 cities in organised order
correct order: A B D C
genes:
    1. probability of step forward on OX [5,7]
    2. probability of step backward on OX [1,3]
    3. probability of step forward on OY [5,7]
    4. probability of step backward on OY [1,3]
'''

class Vendor:
    def __init__(self, vendor_index: int) -> None:
        self.genes: list[int] = []
        self.steps: int = 0
        self.xpos: int = 0
        self.ypos: int = 0
        self.path_history: dict = {}
        #self.population_number: int = population
        self.vendor_index: int = vendor_index
        #self.ready_to_travel: bool
        self.check_point: tuple = (0, 0, 0, {})
        pass

    def __str__(self) -> str:
        return f'Vendor number: {self.vendor_index}\tgenes: {self.genes}'

        
    def genes_init(self, probabilities: list) -> None:
        for probability in probabilities:
            self.genes.append(probability)

    '''def check_genes(self):
        for i in range(4):
            if type(self.genes[i]) is not int:
                print(f'Gene number {i} in Vendor number {self.vendor_index} is not correct')
                self.ready_to_travel = False
            else:
                self.ready_to_travel = True'''
        

    def step_OX(self) -> None:
        choice_of_step = random.randint(1, self.genes[0]+self.genes[1])
        self.steps += 1

        if 1 <= choice_of_step <= self.genes[0]:
            self.xpos += 1
        else:
            self.xpos -= 1

        self.path_history[self.steps] = (self.xpos, self.ypos)

    def step_OY(self) -> None:
        choice_of_step = random.randint(1, self.genes[2]+self.genes[3])
        self.steps += 1

        if 1 <= choice_of_step <= self.genes[2]:
            self.ypos += 1
        else:
            self.ypos -= 1
            
        self.path_history[self.steps] = (self.xpos, self.ypos)
        

    def walk(self) -> None:
        decision = random.randint(0,1)

        if decision == 0:
            self.step_OX()
        else:
            self.step_OY()

    def start_over(self) -> None:
        self.steps = 0
        self.xpos = 0
        self.ypos = 0
        self.path_history = {}

    def go_to_check_point(self) -> None:
        self.steps = self.check_point[0]
        self.xpos = self.check_point[1]
        self.ypos = self.check_point[2]
        self.path_history = self.check_point[3]

    def go_to(self, where_to_go: tuple) -> None:
        while (self.xpos, self.ypos) != where_to_go:
            #self.show_actual_parameters(ranking, start_time, max_times, actual_time)
            if (self.xpos, self.ypos) == where_to_go:
                self.check_point = (self.steps, self.xpos, self.ypos, self.path_history)
                break
            self.walk()
            if self.xpos > 1.1 * where_to_go[0] or self.ypos > 1.1 * where_to_go[1]:
                self.go_to_check_point()
                #break
        #else:
        #    self.check_point = (self.steps, self.xpos, self.ypos, self.path_history)


    '''def show_actual_parameters(self, ranking, start_time, max_times, actual_time):
        clear = lambda: os.system('cls')
        clear()
        sys.stdout.write(f'Testing vendor number: {self.vendor_index}\n')
        sys.stdout.write(f'Actual position: ({self.xpos},{self.ypos})\n')
        sys.stdout.write(f'Ranking: {ranking}\n')
        #sys.stdout.write(f'Path history: {self.path_history}\n')
        sys.stdout.write(f'Time since started: {int(time.time() - start_time)//60} minutes {int(time.time() - start_time)%60} seconds\n')
        if actual_time != 0:
            sys.stdout.write(f'ETA: {round(((max_times - actual_time) * (int(time.time() - start_time)/(actual_time)))/60, 2)} minutes\n')
        else:
            actual_time = 1
            sys.stdout.write(f'ETA: {(max_times/actual_time * (int(time.time() - start_time)/(actual_time*60)))/60} minutes\n')
        sys.stdout.flush()'''
            


    
    