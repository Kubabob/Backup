from numpy import *
import matplotlib.pyplot as plt
import pandas as pd

class Elip_Structure:
    def __init__(self, theta_i, wave_length, thickness, *args) -> None:
        '''
        We need to specify entry parameters, args are tuples of n and k of given material

        if we want to use function that shows plot of for ex. angle then we can just specify parameter as 0
        '''
        self.refractive_indexes = []
        self.complex_refractive_indexes = []
        
        self.theta_angles = [theta_i*pi/180]

        #add (n, k) to list of refractive indexes
        for arg in args:
            self.refractive_indexes.append(arg)

        #make a list of complex refractive indexes based on n and k
        for i in range(len(self.refractive_indexes)):
            self.complex_refractive_indexes.append(self.N(self.refractive_indexes[i][0], self.refractive_indexes[i][1]))

        #make list of theta angles for each layer
        for i in range(len(self.complex_refractive_indexes)-1):
            self.theta_angles.append(self.theta_j(i=i, j=i+1))

        self.wave_length = wave_length
        self.thickness = thickness

        pass

    def beta(self, thickness = None, wave_length = None, i: int = 0, j: int = 1):
        '''
        Returns beta of jth layer
        '''
        if thickness == None:
            thickness = self.thickness
        if wave_length == None:
            wave_length = self.wave_length
        #Ni = self.complex_refractive_indexes[i]
        Nj = self.complex_refractive_indexes[j]
        theta_j = self.theta_angles[j]
        return ((2*pi*thickness)/wave_length) * Nj * cos(theta_j)
    
    def arg(self, complex_number):
        '''
        Returns argument(angle) of complex number

        originally used to compare if numpy's angle func is any different from math arg

        its not
        '''
        if real(complex_number) > 0:
            return arctan(imag(complex_number) / real(complex_number))
        elif real(complex_number) < 0 and imag(complex_number) >= 0:
            return arctan(imag(complex_number) / real(complex_number)) + pi
        else:
            return arctan(imag(complex_number) / real(complex_number)) - pi
        
    def N(self, n, k):
        '''
        Returns complex refractive index based on n and k

        Note:
            in some cases + and - are changed due to differences in other formulas
        Default:
            -
        '''
        return n - 1j*k 
    
    def theta_j(self, Ni = None, Nj = None, theta_i = None, i: int = 0, j: int = 1):
        '''
        Calculates theta angle of jth layer based on Schnells formula
        N0 sin(theta0) = N1 sin(theta1)
        '''
        if Ni != None and Nj != None and theta_i != None:
            Ni = Ni
            Nj = Nj
            theta_i = theta_i
        elif Ni == None and Nj == None and theta_i == None:
            Ni = self.complex_refractive_indexes[i]
            Nj = self.complex_refractive_indexes[j]
            theta_i = self.theta_angles[i]
        else:
            raise ValueError
        return arcsin((Ni/Nj) * sin(theta_i)) 
    
    #not useful
    
    '''def r_p(self, Ni, Nj, theta_i):
        Nji = Nj / Ni
        nominator = ((Nji**2) * cos(theta_i) - sqrt((Nji**2) - sin(theta_i)**2))
        denominator = ((Nji**2) * cos(theta_i) + sqrt((Nji**2) - sin(theta_i)**2))

        return nominator / denominator
    
    def r_s(self, Ni, Nj, theta_i):
        Nji = Nj / Ni
        nominator = sqrt(cos(theta_i) - ((Nji**2) - sin(theta_i)**2))
        denominator = sqrt(cos(theta_i) + ((Nji**2) - sin(theta_i)**2))
        
        return nominator / denominator'''
    
    def r_ij_p(self, i: int = 0, j: int = 1, theta_i: int = None):
        '''
        Returns p- reflectance of 2 layers
        Fresnells formula
        '''
        Ni = self.complex_refractive_indexes[i]
        Nj = self.complex_refractive_indexes[j]
        if theta_i != None:
            theta_i = theta_i
            theta_j = self.theta_j(Ni, Nj, theta_i)
        else:
            theta_i = self.theta_angles[i]
            theta_j = self.theta_angles[j]

        nominator = (Nj * cos(theta_i) - Ni * cos(theta_j))
        denominator = (Nj * cos(theta_i) + Ni * cos(theta_j))

        return nominator / denominator
    
    def r_ij_s(self, i: int = 0, j: int = 1, theta_i: int = None):
        '''
        Returns s- reflectance of 2 layers
        Fresnells formula
        '''
        
        Ni = self.complex_refractive_indexes[i]
        Nj = self.complex_refractive_indexes[j]
        if theta_i != None:
            theta_i = theta_i
            theta_j = self.theta_j(Ni, Nj, theta_i)
        else:
            theta_i = self.theta_angles[i]
            theta_j = self.theta_angles[j]

        nominator = (Ni * cos(theta_i) - Nj * cos(theta_j))
        denominator = (Ni * cos(theta_i) + Nj * cos(theta_j))

        return nominator / denominator
    
    def R_p(self, r_p):
        '''
        Returns p- reflectance but in a representative/comparative form rather than calculational
        '''
        return abs(r_p)**2
    
    def R_s(self, r_s):
        '''
        Returns s- reflectance but in a representative/comparative form rather than calculational
        '''
        return abs(r_s)**2

    def R_n(self, R_p, R_s):
        '''
        Returns mean value of p- and s- comparative reflectances
        
        Shown on refractiveindex.info so maybe somewhat useful
        '''
        return (R_p + R_s)/2
    
    def tan_psi_exp_mdeltai(self, psi = None, delta = None, r_p = None, r_s = None):
        '''
        Probably the most important formula here calculating rho which is tan(psi) * exp(-1i*delta)
        '''
        if psi != None and delta != None:
            return tan(psi) * exp(-1j * delta)
        elif r_p != None and r_s != None:
            return r_p / r_s
        elif r_p == None and r_s == None and psi == None and delta == None:
            r_p = self.r_ij_p()
            r_s = self.r_ij_s()
            return r_p / r_s
        else: raise ValueError

    def r_ijk_p(self, i: int = 0, j: int = 1, k: int = 2, theta_i: int = None, thickness: float = None, wave_length: float = None):
        '''
        Returns p- reflectance of 3 layer structure
        '''
        if theta_i == None:
            r_ij_p = self.r_ij_p(i=i, j=j)
            r_jk_p = self.r_ij_p(i=j,j=k)
        else:
            r_ij_p = self.r_ij_p(i, j, theta_i)
            r_jk_p = self.r_ij_p(j, k, theta_i)

        if thickness != None and wave_length != None:
            beta = self.beta(i=i, j=j, thickness=thickness, wave_length=wave_length)
        elif thickness != None and wave_length == None:
            beta = self.beta(i=i, j=j, thickness=thickness)
        elif thickness == None and wave_length != None:
            beta = self.beta(i=i, j=j, wave_length=wave_length)
        else:
            beta = self.beta(i=i, j=j)

        nominator = (r_ij_p + r_jk_p * exp(-1j*2*beta))
        denominator = (1 + r_ij_p * r_jk_p * exp(-1j*2*beta))

        if denominator != 0:
            return nominator / denominator
        else:
            #print('Division by 0')
            pass

    def r_ijk_s(self, i: int = 0, j: int = 1, k: int = 2, theta_i: int = None, thickness: float = None, wave_length: float = None):
        '''
        Returns s- reflectance of 3 layer structure
        '''
        
        if theta_i == None:
            r_ij_s = self.r_ij_s(i=i, j=j)
            r_jk_s = self.r_ij_s(i=j,j=k)
        else:
            r_ij_s = self.r_ij_s(i, j, theta_i)
            r_jk_s = self.r_ij_s(j, k, theta_i)
        
        if thickness != None and wave_length != None:
            beta = self.beta(i=i, j=j, thickness=thickness, wave_length=wave_length)
        elif thickness != None and wave_length == None:
            beta = self.beta(i=i, j=j, thickness=thickness)
        elif thickness == None and wave_length != None:
            beta = self.beta(i=i, j=j, wave_length=wave_length)
        else:
            beta = self.beta(i=i, j=j)

        nominator = (r_ij_s + r_jk_s * exp(-1j*2*beta))
        denominator = (1 + r_ij_s * r_jk_s * exp(-1j*2*beta))

        return nominator / denominator
    
    def psi(self,layers: int = 2, r_p = None, r_s = None):
        '''
        Returns psi in radians
        '''
        if layers == 2:
            if r_p == None:
                r_p = self.r_ij_p()
            if r_s == None:
                r_s = self.r_ij_s()
        '''elif layers == 3:
            if r_p == None:
                r_p = self.r_ijk_p()'''
        #print(abs(r_p) ,abs(r_s))
        return arctan(abs(r_p) / abs(r_s)) 
    
    def delta(self, r_p = None, r_s = None):
        '''
        Returns delta in radians
        '''
        if r_p == None:
            r_p = self.r_ij_p()
        if r_s == None:
            r_s = self.r_ij_s()
        

        delta = angle(r_p/r_s)
        if delta < 0:
            return 2*pi + delta
        else:
            return delta
    
    def reflectance_plot(self, layers: int = 2):
        '''
        Shows p- s- and mean reflectance plots for choosen structure

        Default 2 top layers
        '''
        x = linspace(0, 90, num=900)
        if layers == 2:
            y_p = [self.R_p(self.r_ij_p(theta_i=(angle*pi/180))) for angle in x]
            y_s = [self.R_s(self.r_ij_s(theta_i=(angle*pi/180))) for angle in x]
            y_n = [self.R_n(y_p[i], y_s[i]) for i in range(len(x))]
        elif layers == 3:
            y_p = [self.R_p(self.r_ijk_p(theta_i=(angle*pi/180))) for angle in x]
            y_s = [self.R_s(self.r_ijk_s(theta_i=(angle*pi/180))) for angle in x]
            y_n = [self.R_n(y_p[i], y_s[i]) for i in range(len(x))]
        else:
            raise ValueError

        plt.plot(x, y_p)
        plt.plot(x, y_s)
        plt.plot(x, y_n)
        plt.grid()
        plt.show()

    def psi_delta_plot(self, layers: int = 2, is_thickness: bool = False, thickness_range: tuple = (0, 1), is_wave_length: bool = False, wave_length_range: tuple = (1, 2000)):
        '''
        Returns plot of psi and delta of given structure

        if we have 3 layer structure we need to decide whether we want plot of thickness or wave length
        '''
        if layers == 2:
            x = linspace(0, 90, num=900)
            y_psi = [rad2deg(self.psi(r_p=self.r_ij_p(theta_i=angle*pi/180), r_s=self.r_ij_s(theta_i=angle*pi/180))) for angle in x]
            y_delta = [rad2deg(self.delta(r_p=self.r_ij_p(theta_i=angle*pi/180), r_s=self.r_ij_s(theta_i=angle*pi/180))) for angle in x]
        elif layers == 3 and is_thickness:
            x = linspace(thickness_range[0], thickness_range[1], num=thickness_range[1]*1000)
            y_psi = [rad2deg(self.psi(r_p=self.r_ijk_p(thickness=thickness), r_s=self.r_ijk_s(thickness=thickness))) for thickness in x]
            y_delta = [rad2deg(self.delta(r_p=self.r_ijk_p(thickness=thickness), r_s=self.r_ijk_s(thickness=thickness))) for thickness in x]
        elif layers == 3 and is_wave_length:
            x = linspace(wave_length_range[0], wave_length_range[1], num=wave_length_range[1])
            y_psi = [rad2deg(self.psi(r_p=self.r_ijk_p(wave_length=wave_length), r_s=self.r_ijk_s(wave_length=wave_length))) for wave_length in x]
            y_delta = [rad2deg(self.delta(r_p=self.r_ijk_p(wave_length=wave_length), r_s=self.r_ijk_s(wave_length=wave_length))) for wave_length in x]
        else:
            raise ValueError

        plt.plot(x, y_psi)
        plt.plot(x, y_delta)
        plt.grid()
        plt.show()

    '''def wave_length_spectrum(self):
        Si = pd.read_csv('Si_nk.csv', sep=',')
        SiO2 = pd.read_csv('SiO2_nk.csv', sep=',')

        psis = {}
        deltas = {}
        for index, wave_length in enumerate(SiO2['wvl']):
            #structure = Elip_Structure(70, 0, 0.7, (1,0), (float(SiO2['n'][index]), 0), (float(Si['n'][index]), float(Si['k'][index])))
            
            self.complex_refractive_indexes = [(1,0), (float(SiO2['n'][index]), 0), (float(Si['n'][index]), float(Si['k'][index]))]
            psis[wave_length] = self.psi(r_p=self.r_ijk_p(wave_length=wave_length), r_s=self.r_ijk_s(wave_length=wave_length))   
            deltas[wave_length] = self.delta(r_p=self.r_ijk_p(wave_length=wave_length), r_s=self.r_ijk_s(wave_length=wave_length))
        else:
            plt.plot(psis.keys(), psis.values())
            plt.plot(deltas.keys(), deltas.values())
            plt.grid()
            plt.show()'''


def get_thickness(angle: int = 0):
    '''
    Doesnt work, probably wrong model psis and deltas
    '''
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

#get_thickness(angle=1)
