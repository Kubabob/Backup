from vendor import Vendor
import time


'''
We need to make population so define our models

genes:
    5. probability of step forward on OX [5,7]
    6. probability of step backward on OX [1,3]
    7. probability of step forward on OY [5,7]
    8. probability of step backward on OY [1,3]

cities cant be the same, so only 21 combinations of first 4 genes

we cross 1st and 2nd, 3rd and 4th and on of the model ranking of the liest steps
'''

class Population:
    def __init__(self, population_number: int, 
                 probability_forward_OX: tuple = (5,6,7),
                 probability_backward_OX: tuple = (1, 2, 3),
                 probability_forward_OY: tuple = (5,6,7),
                 probability_backward_OY: tuple = (1, 2, 3)) -> None:
        
        self.population_number: int = population_number
        self.vendors: list[Vendor] = []
        self.vendor_ranking: dict = {}
        self.best_of_population: list[Vendor] = []

        self.probability_forward_OX = probability_forward_OX
        self.probability_backward_OX = probability_backward_OX
        self.probability_forward_OY = probability_forward_OY
        self.probability_backward_OY = probability_backward_OY
        pass

    def create_population(self):
        if self.population_number == 0:
            for index in range((len(self.probability_forward_OX)**4)):
                self.vendors.append(Vendor(index))
        else:
            for index in range(int(((len(self.probability_forward_OX)**4)//4))*4):
                self.vendors.append(Vendor(index))

        #define genes
        list_of_genes: list[list] = []
        for pfx in self.probability_forward_OX:
            for pbx in self.probability_backward_OX:
                for pfy in self.probability_forward_OY:
                    for pby in self.probability_backward_OY:
                        list_of_genes.append([pfx,pbx,pfy,pby])

        for index, vendor in enumerate(self.vendors):
            vendor.genes_init(list_of_genes[index])

        

    def recombine(self, best_vendor_genes: list[Vendor]):
        for i in range(0, int(len(best_vendor_genes)), 2):
            self.vendors.append(Vendor(vendor_index=len(self.vendors)))
            self.vendors[len(self.vendors)-1].genes_init([best_vendor_genes[i].genes[0], best_vendor_genes[i].genes[1], best_vendor_genes[i+1].genes[2], best_vendor_genes[i+1].genes[3]])
            
            self.vendors.append(Vendor(vendor_index=len(self.vendors)))
            self.vendors[len(self.vendors)-1].genes_init([best_vendor_genes[i+1].genes[0], best_vendor_genes[i+1].genes[1], best_vendor_genes[i].genes[2], best_vendor_genes[i].genes[3]])
            
            self.vendors.append(Vendor(vendor_index=len(self.vendors)))
            self.vendors[len(self.vendors)-1].genes_init([best_vendor_genes[i].genes[0], best_vendor_genes[i].genes[1], best_vendor_genes[i].genes[2], best_vendor_genes[i].genes[3]])
            
            self.vendors.append(Vendor(vendor_index=len(self.vendors)))
            self.vendors[len(self.vendors)-1].genes_init([best_vendor_genes[i+1].genes[0], best_vendor_genes[i+1].genes[1], best_vendor_genes[i+1].genes[2], best_vendor_genes[i+1].genes[3]])



    def test_population(self, test_cities: list[tuple], how_many_times: int, how_many_bests: int):
        start_time = time.time()
        for vendor in self.vendors:
            vendor_steps = 0
            vendor_mean_steps = 0
            for i in range(how_many_times):
                vendor.start_over()
                for city in test_cities:
                    vendor.go_to(city)
                    vendor_steps += vendor.steps
                    
            else:
                vendor_mean_steps = int(vendor_steps/how_many_times)
                self.vendor_ranking[vendor.vendor_index] = vendor_mean_steps
        else:
            self.sort_ranking()            
            for j in range(how_many_bests):
                self.best_of_population.append(self.vendors[list(self.vendor_ranking.keys())[j]])
            


    def sort_ranking(self):
        sorted_vendors = sorted(self.vendor_ranking.items(), key=lambda x:x[1])
        self.vendor_ranking = dict(sorted_vendors)
                    

