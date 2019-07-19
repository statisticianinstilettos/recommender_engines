import numpy as np
import scipy.stats as st 

class HypothesisTests():
    def __init__(self, alpha):
        self.alpha = alpha
        self.z_power = 0.84 #assumes value of 80%
        self.z_critical_upper = st.norm.ppf(1-self.alpha)
        self.z_critical_lower = -st.norm.ppf(1-self.alpha)
        self.z_critical_two_sided = st.norm.ppf(1-self.alpha/2)

        
class TwoPorportionsTest(HypothesisTests):
    '''
    p1 is the test group.
    p2 is the control group.
    
    Significance level of 100(1-alpha)%
    
    Two Tailed Test
    H0: p1-p2 = 0
    HA: p1-p2 =/= 0
    
    Right Tailed Test
    H0: p1-p2 = 0
    HA: p1-p2 > 0
    
    Left Tailed Test
    H0: p1-p2 = 0
    HA: p1-p2 < 0
    
    '''
    def __init__(self, alpha):
        super().__init__(alpha)
        
    def get_sample_size(self, p_control, min_lift, r, tail):
        ''' Calculate sample size for test and control groups'''
        
        if tail == "upper":
            z_critical = self.z_critical_upper
        elif tail == "lower":
            z_critical = self.z_critical_lower 
        elif tail == "two sided":
            z_critical = self.z_critical_two_sided
            
        p2 = p_control
        p1 = p2 + min_lift
        delta = abs(p2-p1)
        pbar = (p1+r*p2)/(r+1)
        qbar = 1-pbar

        m = np.power((z_critical*np.sqrt((r+1)*pbar*qbar) + self.z_power*np.sqrt(r*p1*(1-p1) + p2*(1-p2))),2)/(r*np.power(delta,2))
        n1 = (m/4) * np.power(1+np.sqrt(1+(2*(r+1))/(r*m*delta)), 2)
        N = (r+1)*n1
        n2 = r*n1
        
        print("n1 =",int(np.round(n1)))
        print("n2 =", int(np.round(n2)))
        print("Total sample size =", int(np.round(N)))
        


    def upper_tail_hypothesis_test(self, p1, p2, n1, n2):
        '''
        H0: p1-p2 = 0
        HA: p1-p2 > 0
        Significance level of 100(1-alpha)%
        '''

        z_critical = self.z_critical_upper

        z = np.round((p1-p2)/np.sqrt((p1*(1-p1)/n1)+((p2*(1-p2)/n2))),4)

        if z >= z_critical:
            print("test statistic z:", z)
            print("critical value z*:", z_critical)
            print("Reject H0 in favor of Ha")
        else:
            print("Cannot reject H0")

        p_value = (1 - st.norm.cdf(z))

        print("p-value:", p_value)


    def confidence_interval(self, p1, p2, n1, n2):
        ''' 100(1-alpha)% CI for (p1-p2)'''

        z_critical = self.z_critical_two_sided

        se = np.sqrt((p1*(1-p1)/n1)+((p2*(1-p2)/n2)))
        upper = np.round((p1-p2) + z_critical*se, 4)
        lower = np.round((p1-p2) - z_critical*se, 4)

        print("confidence interval:", lower, ">= (p1-p2) <=", upper)
