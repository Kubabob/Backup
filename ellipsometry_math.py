from numpy import *
import matplotlib.pyplot as plt

class Ellipsometry:
    def __init__(self, theta_i, wave_length, thickness, *args) -> None:
        self.refractive_indexes = []
        self.complex_refractive_indexes = []
        
        self.betas = []

        for arg in args:
            self.refractive_indexes.append(arg)

        for i in range(len(self.refractive_indexes)):
            self.complex_refractive_indexes.append(self.N(self.refractive_indexes[i][0], self.refractive_indexes[i][1]))

        self.theta_angles = [theta_i * pi / 180, self.theta_j(i=0, j=1)]
        '''for i in range(1, len(self.complex_refractive_indexes)-1):
            self.theta_angles.append(self.theta_j(i=i, j=i+1))
            #self.betas.append(self.beta())
        else: print(self.theta_angles)'''

        #self.theta_angles[0] = theta_i * pi / 180
        self.wave_length = wave_length
        self.thickness = thickness

        pass

    def beta(self, thickness = None, wave_length = None, Ni = None, Nj = None, theta_i = None, i: int = 0, j: int = 1):
        if thickness == None:
            thickness = self.thickness
        if wave_length == None:
            wave_length = self.wave_length
        '''if Ni == None:
            Ni = self.complex_refractive_indexes[0]
        if Nj == None:
            Nj = self.complex_refractive_indexes[1]
        if theta_i == None:
            theta_i = self.theta_j()'''
        Ni = self.complex_refractive_indexes[i]
        Nj = self.complex_refractive_indexes[j]
        theta_i = self.theta_angles[i]
        return ((2*pi*thickness)/wave_length) * sqrt((Nj**2) - (Ni**2)*sin(theta_i)**2)
    
    def arg(self, complex_number):
        if real(complex_number) > 0:
            return arctan(imag(complex_number) / real(complex_number))
        else:
            return arctan(imag(complex_number) / real(complex_number)) + pi
        
    def N(self, n, k):
        return n - 1j*k
    
    def theta_j(self, Ni = None, Nj = None, theta_i = None, i: int = 0, j: int = 1):
        '''if Ni == None:
            Ni = self.complex_refractive_indexes[0]
        if Nj == None:
            Nj = self.complex_refractive_indexes[1]
        if theta_i == None:
            theta_i = self.theta_angles[0]'''
        Ni = self.complex_refractive_indexes[i]
        Nj = self.complex_refractive_indexes[j]
        theta_i = self.theta_angles[i]
        return arcsin((Ni * sin(theta_i)) / Nj) * pi / 180
    
    def r_p(self, Ni, Nj, theta_i):
        Nji = Nj / Ni
        nominator = ((Nji**2) * cos(theta_i) - sqrt((Nji**2) - sin(theta_i)**2))
        denominator = ((Nji**2) * cos(theta_i) + sqrt((Nji**2) - sin(theta_i)**2))

        return nominator / denominator
    
    def r_s(self, Ni, Nj, theta_i):
        Nji = Nj / Ni
        nominator = sqrt(cos(theta_i) - ((Nji**2) - sin(theta_i)**2))
        denominator = sqrt(cos(theta_i) + ((Nji**2) - sin(theta_i)**2))
        
        return nominator / denominator
    
    def r_ij_p(self, Ni = None, Nj = None, theta_i = None, theta_j = None, i: int = 0, j: int = 1):
        '''if Ni == None:
            Ni = self.complex_refractive_indexes[0]
        if Nj == None:
            Nj = self.complex_refractive_indexes[1]
        if theta_i == None:
            theta_i = self.theta_angles[0]
        if theta_j == None:
            theta_j = self.theta_j(Ni, Nj, theta_i)'''
        Ni = self.complex_refractive_indexes[i]
        Nj = self.complex_refractive_indexes[j]
        theta_i = self.theta_angles[i]
        theta_j = self.theta_angles[j]

        nominator = (Nj * cos(theta_i) - Ni * cos(theta_j))
        denominator = (Nj * cos(theta_i) + Ni * cos(theta_j))

        self.r_01_p = nominator / denominator

        return self.r_01_p
    
    def r_ij_s(self, Ni = None, Nj = None, theta_i = None, theta_j = None, i: int = 0, j: int = 1):
        '''if Ni == None:
            Ni = self.complex_refractive_indexes[0]
        if Nj == None:
            Nj = self.complex_refractive_indexes[1]
        if theta_i == None:
            theta_i = self.theta_angles[0]
        if theta_j == None:
            theta_j = self.theta_j(Ni, Nj, theta_i)'''
        
        Ni = self.complex_refractive_indexes[i]
        Nj = self.complex_refractive_indexes[j]
        theta_i = self.theta_angles[i]
        theta_j = self.theta_angles[j]

        nominator = (Ni * cos(theta_i) - Nj * cos(theta_j))
        denominator = (Ni * cos(theta_i) + Nj * cos(theta_j))

        self.r_01_s = nominator / denominator

        return self.r_01_s
    
    def R_p(self, r_p):
        return abs(r_p)**2
    
    def R_s(self, r_s):
        return abs(r_s)**2

    def R_n(self, R_p, R_s):
        return (R_p + R_s)/2
    
    def tan_psi_exp_mdeltai(self, psi = None, delta = None, r_p = None, r_s = None):
        if psi != None and delta != None:
            return tan(psi) * exp(1j * delta)
        elif r_p != None and r_s != None:
            return r_p / r_s
        elif r_p == None and r_s == None and psi == None and delta == None:
            r_p = self.r_ij_p()
            r_s = self.r_ij_s()
            return r_p / r_s
        else: raise ValueError

    def r_ijk_p(self, r_ij_p = None, r_jk_p = None, beta = None, i: int = 0, j: int = 1, k: int = 2):
        '''if r_ij_p == None:
            r_ij_p = self.r_ij_p()
        if r_jk_p == None:
            r_jk_p = self.r_ij_p(self.complex_refractive_indexes[1],
                                 self.complex_refractive_indexes[2],
                                 self.theta_j(self.complex_refractive_indexes[0],
                                              self.complex_refractive_indexes[1]))
        if beta == None:
            beta = self.beta()'''
        
        r_ij_p = self.r_ij_p(i=i, j=j)
        r_jk_p = self.r_ij_p(i=j,j=k)
        beta = self.beta(i=i, j=j)

        nominator = (r_ij_p + r_jk_p * exp(-2j*beta))
        denominator = (1 + r_ij_p * r_jk_p * exp(-2j*beta))

        return nominator / denominator
    
    def r_ijk_s(self, r_ij_s = None, r_jk_s = None, beta = None, i: int = 0, j: int = 1, k: int = 2):
        '''if r_ij_s == None:
            r_ij_s = self.r_ij_s()
        if r_jk_s == None:
            r_jk_s = self.r_ij_s(self.complex_refractive_indexes[1],
                                 self.complex_refractive_indexes[2],
                                 self.theta_j(self.complex_refractive_indexes[0],
                                              self.complex_refractive_indexes[1]))
        if beta == None:
            beta = self.beta()'''
        
        r_ij_s = self.r_ij_s(i=i, j=j)
        r_jk_s = self.r_ij_s(i=j, j=k)
        beta = self.beta(i=i, j=j)

        nominator = (r_ij_s + r_jk_s * exp(-2j*beta))
        denominator = (1 + r_ij_s * r_jk_s * exp(-2j*beta))

        return nominator / denominator
    
    def psi(self, r_p = None, r_s = None):
        if r_p == None:
            r_p = self.r_ij_p()
        if r_s == None:
            r_s = self.r_ij_s()
        return arctan(abs(r_p) / abs(r_s))
    
    def delta(self, r_p = None, r_s = None):
        if r_p == None:
            r_p = self.r_ij_p()
        if r_s == None:
            r_s = self.r_ij_s()
        
        return self.arg(r_p) - self.arg(r_s)
    
    def reflectance_plot(self):
        x = linspace(0, 90, num=900)
        y_p = [self.R_p(self.r_ij_p(theta_i=(angle*pi/180))) for angle in x]
        y_s = [self.R_s(self.r_ij_s(theta_i=(angle*pi/180))) for angle in x]
        y_n = [self.R_n(y_p[i], y_s[i]) for i in range(len(x))]

        plt.plot(x, y_p)
        plt.plot(x, y_s)
        plt.plot(x, y_n)
        plt.show()

'''Si_6328 = (3.8827, 0.019626)
Air_6328 = (1.00027653, 0)
SiO2_6328 = (1.4570, 0)'''

'''E_model = Ellipsometry(70, 0.6328, Air_6328, Si_6328)

#E_model.reflectance_plot()

model_value = E_model.tan_psi_exp_mdeltai()

print(model_value)
'''