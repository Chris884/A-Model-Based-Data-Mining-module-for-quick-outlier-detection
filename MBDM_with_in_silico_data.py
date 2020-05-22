import os
import sys
from scipy.optimize import minimize, basinhopping
import multiprocessing as multi
import time
from MBDM_with_in_silico_data_functions import *

start_time = time.time()

abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)

def powerlaw(n,L,U,theta):
    
    # mole fraction for each species
    (CH_4,O_2,CO_2,H_2O) = n
    
    #(k_1, Ea) = theta
    (k_1, Ea) = theta
    
    # Design variables
    (u,T,Pin,Pout) = U                             #mL/min, Celsius
    
    #stoichiometry
    nu = np.array([-1,-2,+1,+2])
    
    # rate constant & Arrhenius equation
    k = np.exp(-k_1 - (Ea*10000.0/R) * ((1/(T+273.15))-(1/(Tm+273.15))))
    
    Pavg = (Pin + Pout)/2
    
    # rate expression
    rate = k * Pavg * CH_4
    
    # for gas phase reactions in packed bed reactor (this is a conversion factor)
    alpha = ((R*(T+273.15))/(Pavg*1e5*(u*(1e-6/60)*(P0/Pavg)*((T+273.15)/(T0+273.15)))))
    
    # differential mole balance
    dndt = np.array(np.dot(np.transpose(nu),rate*alpha))
    
    return dndt
    
def MVK(n,L,U,theta):
    
    # mole fraction for each species
    (CH_4,O_2,CO_2,H_2O) = n
    
    #Design variables
    (u,T,Pin,Pout) = U                                                         #mL/min, Celsius

    (k_1,Ea_1,k_2,Ea_2,k_3,Ea_3) = theta

    #stoichiometry
    nu = np.array([-1,-2,+1,+2])

    #kinetic expressions
    k1 = np.exp(-k_1-((Ea_1*10000.0/R)*((1/(T+273.15))-(1/(Tm+273.15)))))   #mol/bar.min.g
    k2 = np.exp(-k_2-((Ea_2*10000.0/R)*((1/(T+273.15))-(1/(Tm+273.15)))))   #mol/bar.min.g
    k3 = np.exp(-k_3-((Ea_3*10000.0/R)*((1/(T+273.15))-(1/(Tm+273.15)))))   #mol/min.g

    #Definitions
    Pavg = (Pin + Pout)/2
    P_O_2, P_CH_4 = O_2*Pavg, CH_4*Pavg
    
    # rate expression
    rate = ((k1*k2*P_CH_4*P_O_2)/(k1*P_O_2+2*k2*P_CH_4+((k1*k2)/k3)*P_O_2*P_CH_4))     #mol/g.s
    
    # for gas phase reactions in packed bed reactor (this is a conversion factor)
    alpha = ((R*(T+273.15))/(Pavg*1e5*(u*(1e-6/60)*(P0/Pavg)*((T+273.15)/(T0+273.15))))) #s/mol
    
    # differential mole balance
    dndt = np.array(np.dot(np.transpose(nu),rate*alpha))
    
    return dndt

def LH(n,L,U,theta):
    
    # mole fraction for each species
    (CH_4,O_2,CO_2,H_2O) = n
    
    #Design variables
    (u,T,Pin,Pout) = U                                                           #mL/min, Celsius

    (k_1,H_1,k_2,H_2,k_3,H_3) = theta

    #stoichiometry
    nu = np.array([-1,-2,+1,+2])

    #kinetic expressions
    ksr = np.exp(-k_1-((H_1*10000.0/R)*((1/(T+273.15))-(1/(Tm+273.15)))))   #mol/bar.min.g
    Ko = np.exp(k_2-((H_2*10000.0/R)*((1/(T+273.15))-(1/(Tm+273.15)))))   #1/bar
    Km = np.exp(k_3-((H_3*10000.0/R)*((1/(T+273.15))-(1/(Tm+273.15)))))   #1/bar

    #Definitions
    Pavg = (Pin + Pout)/2
    P_O_2, P_CH_4 = O_2*Pavg, CH_4*Pavg

    # rate expression
    rate = ((ksr*Ko*Km*P_O_2*P_CH_4)/((1+(Km*P_CH_4)+(Ko*P_O_2))**2))  #mol/g.s
    
    # for gas phase reactions in packed bed reactor (this is a conversion factor)
    alpha = ((R*(T+273.15))/(Pavg*1e5*(u*(1e-6/60)*(P0/Pavg)*((T+273.15)/(T0+273.15))))) #s/mol
    
    # differential mole balance
    dndt = np.array(np.dot(np.transpose(nu),rate*alpha))
    
    return dndt

def full_factorial(model, theta, design_space, levels, sigma, measurable):
    
    #Generation of designs
    (v_range, T_range)=design_space
    (T_levels,v_levels)=levels
    
    T,v=np.mgrid[T_range[0]:T_range[1]:np.complex(0,T_levels),v_range[0]:v_range[1]:np.complex(0,v_levels)]
    
    dataset = np.array(np.prod(levels))
    dataset = experiment(n0,[v.ravel()[0],T.ravel()[0],Pin,Pout],theta,[mc],sigma,model,measurable)

    for i in range(1,np.prod(T.shape)):
        dataset=np.append(dataset,experiment(n0,[v.ravel()[i],T.ravel()[i],Pin,Pout],theta,[mc],sigma,model,measurable),axis=0)
    
    return dataset

##### Degrees of freedom #####

mc=0.01                                       # g
n0 = np.array([0.025,0.1,0.0,0.0])            # mol fraction
sigma=0.002*np.ones(len(n0))                  # standard deviation of in-silico data points
measurable = np.array([0,1,2,3])              # component pooled standard deviation
R = 8.314                                     # J/mol.K
T0, Tm = 20.0, 320.0                          # Celsius
P0, Pin, Pout = 1.0,  2.0, 1.0                # bar

######## Generation of design values ########

T_range=(250.0,350.0)                         # Celsius
v_range=(20,30)                               # ml/min
design_space=(v_range, T_range)               # design space for factorial design
no = 4                                        # number of levels
levels=(no,no)                                # sampling points

### True kinetic parameters ###

true_theta_powerlaw = [7.1,9.4]
true_theta_LH  = [ 9.1 , 8.0 , 3.8 , 0.029 , 4.5 , 0.016 ]
true_theta_MVK = [ 1.5 , 8.8 , 6.8 , 6.2 , 10.0 , 9.4 ]

### Initial guesses ###

initial_guess_powerlaw = np.array([7,9])
initial_guess_LH  = np.array([ 8 , 10 , 5 , 0.025 , 3 , 0.02 ])
initial_guess_MVK = np.array([ 1.2 , 10 , 5 , 7.5 , 11 , 8.5 ])

##### Choosing a candidate model #####

models = [powerlaw,LH,MVK]
theta_models = [true_theta_powerlaw,true_theta_LH,true_theta_MVK]
initial_guesses = [initial_guess_powerlaw,initial_guess_LH,initial_guess_MVK]

candidate_model = models[0]
initial_guess = initial_guesses[0]
true_model = models[0]
true_theta = theta_models[0]

####### Generation of preliminary design #######

np.random.seed(seed=7)
dataset = full_factorial(true_model, true_theta, design_space, levels, sigma, measurable)

####### Artificial perturbations #######

### Exp 1 ###
np.put(dataset[0][3],[0],dataset[0][3][0]*0.5)
np.put(dataset[0][3],[3],dataset[0][3][3]*1.3)

### Exp 4 ###
np.put(dataset[3][3],[1],dataset[3][3][1]*0.5)

### Exp 8 ###
np.put(dataset[7][3],[0],dataset[7][3][0]*1.25)

### Exp 10 ###
np.put(dataset[9][3],[2],dataset[9][3][2]*2)

### Exp 16 ###
np.put(dataset[-1][3],[1],dataset[-1][3][1]*0.9)
np.put(dataset[-1][3],[2],dataset[-1][3][2]*0.95)
np.put(dataset[-1][3],[3],dataset[-1][3][3]*1.1)

######## Maximum Likelihood Estimation ##########

up_bound=initial_guess+0.5*initial_guess
low_bound=initial_guess*0.001
bnds=tuple(map(tuple, np.c_[low_bound,up_bound]))

#Phi = minimize(loglikelihood, initial_guess, args=(dataset, sigma, candidate_model, measurable,), method='SLSQP',bounds=bnds,options={'disp': True, 'ftol': 1e-20, 'maxiter':300})
minimizer_kwargs = {"method":"SLSQP","args":(dataset, sigma, candidate_model, measurable,),"bounds":bnds,"options":{'disp': False, 'ftol': 1e-20, 'maxiter':500}}
Phi = basinhopping(loglikelihood, initial_guess, minimizer_kwargs=minimizer_kwargs,niter=15,T=0.1,stepsize=0.1)
print(Phi.fun)
print('MLE kinetic parameters: ', Phi.x)
if candidate_model == powerlaw:
    print('Actual MLE kinetic parameters: ', [np.exp(-Phi.x[0]),Phi.x[1]*(10**4)])
elif candidate_model == MVK:
    print('Actual MLE kinetic parameters: ', [np.exp(-Phi.x[0]),Phi.x[1]*(10**4),np.exp(-Phi.x[2]),
                                              Phi.x[3]*(10**4),np.exp(-Phi.x[4]),Phi.x[5]*(10**4)])
else:
    print('Actual MLE kinetic parameters: ', [np.exp(-Phi.x[0]),Phi.x[1]*(10**4),np.exp(Phi.x[2]),
                                              Phi.x[3]*(10**4),np.exp(Phi.x[4]),Phi.x[5]*(10**4)])
estimated_parameters = Phi.x
MLE_residuals, MLE_prediction, MLE_measurement = MLE_values(estimated_parameters, dataset, sigma, candidate_model, measurable)

####### Inputs to Model-Based Data Mining section #######

tolerance = 1.96

############# Model-Based Data Mining #############

kinetic_parameters, binary_switchers, sample_contribution = MBDM(estimated_parameters, up_bound, low_bound, dataset, sigma, candidate_model, measurable, tolerance)

print('MBDM kinetic parameters: ', kinetic_parameters)
if candidate_model == powerlaw:
    print('Actual MBDM kinetic parameters: ', [np.exp(-kinetic_parameters[0]),kinetic_parameters[1]*(10**4)])
elif candidate_model == MVK:
    print('Actual MBDM kinetic parameters: ', [np.exp(-kinetic_parameters[0]),kinetic_parameters[1]*(10**4),np.exp(-kinetic_parameters[2]),
                                              kinetic_parameters[3]*(10**4),np.exp(-kinetic_parameters[4]),kinetic_parameters[5]*(10**4)])
else:
    print('Actual MBDM kinetic parameters: ', [np.exp(-kinetic_parameters[0]),kinetic_parameters[1]*(10**4),np.exp(kinetic_parameters[2]),
                                              kinetic_parameters[3]*(10**4),np.exp(kinetic_parameters[4]),kinetic_parameters[5]*(10**4)])
print('The estimated switchers are: ', binary_switchers)
print('The contribution of each sample to the objective is: ', sample_contribution)
print('Check if indeed it happens that samples with negative contribution are given switcher -1.')

reduced_dataset = MBDM_dataset(binary_switchers,dataset)

MBDM_residuals, MBDM_prediction, MBDM_measurement = MLE_values(kinetic_parameters, reduced_dataset, sigma, candidate_model, measurable)
MBDM_residuals_all_data, MBDM_prediction_all_data, MBDM_measurement_all_data = MLE_values(kinetic_parameters, dataset, sigma, candidate_model, measurable)
MBDM_resids(MBDM_residuals_all_data,sample_contribution)

#Threshold switchers at 0.5 to make sure that only integer values are fed as labels
#also, the values 0, 1 must be converted into -1 and +1

int_switchers = np.array([-1 if i < 0.5 else 1 for i in binary_switchers])

print(int_switchers)

##### Plots #####

Comp_evolution('Maximum Likelihood Estimator',candidate_model,n0,mc,estimated_parameters)
Comp_evolution('MBDM',candidate_model,n0,mc,kinetic_parameters)

Barplots('MLE experiment dependent residuals',MLE_residuals,int_switchers)
Barplots('MBDM experiment dependent residuals',MBDM_residuals_all_data,int_switchers)

Switchers_bar_plot(binary_switchers)

Parityplot('Maximum Likelihood Estimator',MLE_measurement, MLE_prediction,dataset,tolerance,sigma)
Parityplot('MBDM',MBDM_measurement, MBDM_prediction,reduced_dataset,tolerance,sigma)

Residual_dist('Maximum Likelihood Estimator',MLE_measurement,MLE_prediction,sigma,13)
Residual_dist('MBDM',MBDM_measurement,MBDM_prediction,sigma,13)

Binary_switchers_table(dataset,binary_switchers)
#we also need to provide the ranges for each design variable to normalise the features

# Reliability_map(dataset, int_switchers)

CH4_contribution = CH4_Deviations(int_switchers,MBDM_prediction_all_data,MBDM_measurement_all_data,sigma,tolerance)
O2_contribution = O2_Deviations(int_switchers,MBDM_prediction_all_data,MBDM_measurement_all_data,sigma,tolerance)
CO2_contribution = CO2_Deviations(int_switchers,MBDM_prediction_all_data,MBDM_measurement_all_data,sigma,tolerance)
H2O_contribution = H2O_Deviations(int_switchers,MBDM_prediction_all_data,MBDM_measurement_all_data,sigma,tolerance)
Total_contribution, omitted_experiments = Total_deviation(int_switchers,CH4_contribution,O2_contribution,CO2_contribution,H2O_contribution,sample_contribution)
Heatmap(CH4_contribution,O2_contribution,CO2_contribution,H2O_contribution,Total_contribution,omitted_experiments)

############ Statistical tests ############

##### Chi-square test #####

alpha = 0.05
MLE_n_prelim = len(dataset)
n_y = len(measurable)
n_theta = len(initial_guess)

## Maximum Likelihood Estimator ##

MLE_chi_value = Phi.fun
MLE_chi_test = chisquare_test(alpha,MLE_n_prelim,n_y,n_theta,MLE_chi_value)

if MLE_chi_test[1]>MLE_chi_test[0]:
    print('MLE failed chi-square test: chi2 of the sample',MLE_chi_test[1],'is higher than the chi-square of reference',MLE_chi_test[0])
else:
    print('MLE successful chi-square test: chi2 of the sample',MLE_chi_test[1],'is lower than the chi-square of reference',MLE_chi_test[0])

## MBDM ##

MBDM_n_prelim = len(reduced_dataset)

MBDM_chi_value = sum(sum(MBDM_residuals))
MBDM_chi_test = chisquare_test(alpha,MBDM_n_prelim,n_y,n_theta,MBDM_chi_value)

if MBDM_chi_test[1]>MBDM_chi_test[0]:
    print('MBDM failed chi-square test: chi2 of the sample',MBDM_chi_test[1],'is higher than the chi-square of reference',MBDM_chi_test[0])
else:
    print('MBDM successful chi-square test: chi2 of the sample',MBDM_chi_test[1],'is lower than the chi-square of reference',MBDM_chi_test[0])

## t-test ##

epsilon = 0.001
t = np.linspace(0.0,mc,5)
u = [300.0, 25.0, 2.0, 1.00]

## Maximum Likelihood Estimator ##

MLE_obscovariancematrix = obs_covariance(epsilon,estimated_parameters,n_theta,n_y,MLE_n_prelim,candidate_model,initial_guess,n0,t,sigma,dataset,mc)
MLE_obscorrelationmatrix = correlation(n_theta,MLE_obscovariancematrix)

MLE_confidenceinterval, MLE_t_statistic, MLE_t_ref_val = t_analysis(n_theta,MLE_obscovariancematrix,alpha,MLE_n_prelim,n_y,estimated_parameters)

print('MLE t-test confidence interval:',MLE_confidenceinterval)
print('MLE t-values for estimated parameters:',MLE_t_statistic)
print('MLE t-reference value:',MLE_t_ref_val)

for i in range(0,len(estimated_parameters)):
    if MLE_t_statistic[i]<MLE_t_ref_val:
        print('MLE failed t-test: t-statistic for parameter',i+1,'is less than the t reference value')
    else:
        print('MLE successful t-test: t-statistic for parameter',i+1,'is greater than the t reference value')
        
## MBDM ##

MBDM_obscovariancematrix = obs_covariance(epsilon,kinetic_parameters,n_theta,n_y,MBDM_n_prelim,candidate_model,estimated_parameters,n0,t,sigma,reduced_dataset,mc)
MBDM_obscorrelationmatrix = correlation(n_theta,MBDM_obscovariancematrix)

MBDM_confidenceinterval, MBDM_t_statistic, MBDM_t_ref_val = t_analysis(n_theta,MBDM_obscovariancematrix,alpha,MBDM_n_prelim,n_y,kinetic_parameters)

print('MBDM t-test confidence interval:',MBDM_confidenceinterval)
print('MBDM t-values for estimated parameters:',MBDM_t_statistic)
print('MBDM t-reference value:',MBDM_t_ref_val)

for i in range(0,len(kinetic_parameters)):
    if MBDM_t_statistic[i]<MBDM_t_ref_val:
        print('MBDM failed t-test: t-statistic for parameter',i+1,'is less than the t reference value')
    else:
        print('MBDM successful t-test: t-statistic for parameter',i+1,'is greater than the t reference value')

print("--- %s seconds ---" % (time.time() - start_time))
os.system('afplay /System/Library/Sounds/Sosumi.aiff')