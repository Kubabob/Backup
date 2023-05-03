import random
'''
Genome

1. we define a list of numbers
2. we make random gene that is a list of 0 and 1 that define if number from numbers list count or not
'''

class Genome:
    def __init__(self, index: int, numbers: list, genes: list = []) -> None:
        self.index = index
        self.numbers = numbers
        self.genes = genes
        self.genes_list1: list = []
        self.genes_list2: list = []
        self.length_of_genome = len(self.numbers)
        self.best_possible_sum = sum(self.numbers)/2
        self.genome_fitness: float
        self.mutation_chance: int = random.randint(1,10)

        pass

    def gene_init(self):
        genes = []
        for _ in range(self.length_of_genome):
            genes.append(random.randint(0,1))

        #print(f'Index: {self.index} genes: {genes}')

        self.genes = genes

    def fitness_count(self):
        genes_sum = 0
        for i in range(self.length_of_genome):
            if self.genes[i] == 1:
                genes_sum += self.numbers[i]
                
        self.genome_fitness = 1 - abs(self.best_possible_sum-genes_sum)/self.best_possible_sum

    def mutate(self):
        if self.mutation_chance < 3:
            mutation_point = random.randint(0, len(self.genes)-1)
            if self.genes[mutation_point] == 0:
                self.genes[mutation_point] = 1
            else:
                self.genes[mutation_point] = 0
            

    def __str__(self) -> str:
        for i in range(self.length_of_genome):
            if self.genes[i] == 1:
                self.genes_list1.append(self.numbers[i])
            else:
                self.genes_list2.append(self.numbers[i])

        return f'List 1: {self.genes_list1}\nSum of list 1: {sum(self.genes_list1)}\nList 2: {self.genes_list2}\nSum of list 2: {sum(self.genes_list2)}'
            
    

