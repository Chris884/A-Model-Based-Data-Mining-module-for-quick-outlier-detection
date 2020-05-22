import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import interp1d
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns

###### Importing the necessary data #######

exp_results = pd.read_excel('LatinHyperCDoEResults.xlsx', 'DoE')
data_array = exp_results.values
    
###### Generating the dataset #########

def measurement(data,experiment,t,measurable):
    
    X_inlet_conditions = np.array([data[experiment][4],data[experiment][3]*data[experiment][4],0])
    X_design_variables = np.array([data[experiment][2],data[experiment][1],data[experiment][5]])
    X_outlet_data = np.array([data[experiment][8],data[experiment][7],data[experiment][6]])
    
    return [X_inlet_conditions,X_design_variables,t,X_outlet_data,np.array(measurable)]

def gen_dataset(data,t,measurable):
    
    dataset = np.array([measurement(data,0,t,measurable)])
    for i in range(1,len(data)):
        dataset=np.append(dataset,[measurement(data,i,t,measurable)],axis=0)
        
    return dataset

#### Pressure drop model #####

def Pressure_drop(x,L,U,params):
    P = x[0]
    
    (u,T) = U  
    
    (mu,P0,T0,eps,dp,As,rho0) = params
    
    dPdz = -1.0*(((150.0*mu*(u*(1e-6/60.0)*(P0*1e5/P)*((T+273.15)/(T0+273.15)))*((1-eps)**2))/((dp**2)*As*(eps**3))) + ((1.75*(rho0*((u*(1e-6/60.0))**2)*(P0*1e5/P)*((T+273.15)/(T0+273.15)))*(1-eps))/(dp*(As**2)*(eps**3))))
    return dPdz

def loglikelihood_pdc(c_theta,dataset,L,params,inletpressure_m):
    solutions = []
    for i in range(len(dataset)):
        Pout = odeint(Pressure_drop,dataset[i][1][2]*1e5,np.linspace(L,0.0,5), args=([dataset[i][1][0],dataset[i][1][1]],params,))
        solutions.append(Pout)
    phat_array = np.asarray(solutions)
    phat_measured = np.zeros(len(dataset))
    for i in range(len(dataset)):
        phat_measured[i] = phat_array[i][-1]*1e-5 + c_theta
    residual_array = np.zeros(len(dataset))
    for i in range(len(dataset)):
        residual_array[i] = (inletpressure_m[i] - phat_measured[i])**2
    resum = sum(residual_array[i] for i in range(len(dataset)))
    wt_rho_ind = np.zeros(len(dataset))
    for i in range(len(dataset)):
        wt_rho_ind[i] = (residual_array[i]/(0.005**2))
    weighted_rho_overall = sum(wt_rho_ind[i] for i in range(len(dataset)))
    obj_fun = 0.5 * weighted_rho_overall      # IID with constant variance (measurement covariance matrix is a scalar matrix if the standard deviation is same or a diagonal matrix if it is different)
    return obj_fun
    
def parameter_estimation_pdc(c_theta,dataset,L,params,inletpressure_m):
    new_estimate = minimize_scalar(loglikelihood_pdc,bounds = (1e-6,1e6), method = 'bounded',args=(dataset,L,params,inletpressure_m,))
    #new_estimate = minimize(loglikelihood,param_initialguess2,method = 'SLSQP', bounds = ([1e-30,1e5],[1e-30,1e5],[-1e30,1e5],[1e-30,1e5],[1e-30,1e5],[1e-30,1e5],[-1e30,1e5],[1e-30,1e5],[1e-30,1e5],[1e-30,1e5],[-1e30,1e5],[1e-30,1e5],[1e-30,1e5]), args=(u,))
    return new_estimate
    
def simulation_pdc(c_theta,dataset,c_estimate,L,inletpressure_m,params):
    solutions = []
    for i in range(len(dataset)):
        Pout = odeint(Pressure_drop,dataset[i][1][2]*1e5,np.linspace(L,0.0,5), args=([dataset[i][1][0],dataset[i][1][1]],params,))
        solutions.append(Pout)
    phat_array = np.asarray(solutions)
    phat_measured = np.zeros(len(dataset))
    for i in range(len(dataset)):
        phat_measured[i] = phat_array[i][-1]*1e-5 + c_estimate
    residual_array = np.zeros(len(dataset))
    for i in range(len(dataset)):
        residual_array[i] = (inletpressure_m[i] - phat_measured[i])**2
    return residual_array, phat_measured

def pinletback(dataset,L,inletpressure_m,params):
    result = []
    axial_domain = []
    for i in range(len(dataset)):
        Pout = odeint(Pressure_drop,dataset[i][1][2]*1e5,np.linspace(L,0.0,5), args=([dataset[i][1][0],dataset[i][1][1]],params))
        result.append(Pout)
        axial_domain.append(np.linspace(L,0.0,5))
    Pinback = np.asarray(result)*1e-5
    Pin_p = np.zeros(len(dataset))
    for i in range(len(dataset)):
        Pin_p[i] = Pinback[i][-1]
    res_array = np.zeros(len(dataset))
    for i in range(len(dataset)):
        res_array[i] = (inletpressure_m[i] - Pin_p[i])**2
    res_sum = sum(res_array[i] for i in range(len(dataset)))
    return Pin_p, res_sum

######## Maximum likelihood estimator functions ########

def loglikelihood(theta, dataset, sigma, model, measurable,c_theta,c_estimate,L,inletpressure_m,params):
    
    loglikelihood.residuals = np.zeros(shape=(len(dataset),len(sigma)))
    loglikelihood.predictions=np.zeros(shape=(len(dataset),len(sigma)))
    loglikelihood.measurements=[dataset[i][3] for i in range(0,len(dataset))]
    
    for i in range(0,len(dataset)):
        X=odeint(model,dataset[i][0],[0.0,dataset[i][2]],args=(dataset[i][1][0:2],theta,(((simulation_pdc(c_theta,dataset,c_estimate,L,inletpressure_m,params)[1][i]+dataset[i][1][2])/2.0))))
        loglikelihood.predictions[i]=X[-1]
    
    loglikelihood.residuals =((loglikelihood.measurements-loglikelihood.predictions)/sigma)**2
    objective=0

    for i in measurable:
        objective=objective+np.sum(loglikelihood.residuals[:,i])
    
    return objective

def MLE_values(theta, dataset, sigma, model, measurable,c_theta,c_estimate,L,inletpressure_m,params):
    
    loglikelihood(theta, dataset, sigma, model, measurable,c_theta,c_estimate,L,inletpressure_m,params)
    
    return loglikelihood.residuals, loglikelihood.predictions, loglikelihood.measurements

######## MBDM functions ########

def MBDM(theta, up_bounds, low_bounds, dataset, sigma, model, measurable, tolerance_MBDM,c_theta,c_estimate,L,inletpressure_m,params):
    
    data_mining_parameter_set=np.array(list(theta)+list(np.ones(len(dataset))))
    up_bounds_all=np.array(list(up_bounds)+list(np.ones(len(dataset))))
    low_bounds_all=np.array(list(low_bounds)+list(np.zeros(len(dataset))))
    optimisation_variables_bounds=tuple(map(tuple, np.c_[low_bounds_all,up_bounds_all]))
    
    def MBDM_constraints(optimisation_variables):
    
        parameters=optimisation_variables[0:len(theta)]
        switchers=optimisation_variables[len(theta):]
        residuals=np.zeros(shape=(len(dataset),len(sigma)))
        predictions=np.zeros(shape=(len(dataset),len(sigma)))
        measurements=[dataset[i][3] for i in range(0,len(dataset))]
    
        for i in range(0,len(dataset)):
            X=odeint(model,dataset[i][0],[0.0,dataset[i][2]],(dataset[i][1], parameters,(((simulation_pdc(c_theta,dataset,c_estimate,L,inletpressure_m,params)[1][i]+dataset[i][1][2])/2.0))))
            predictions[i]=X[1]
    
        residuals=((measurements-predictions)/sigma)**2
        residual_term=np.array([np.sum(residuals[i,measurable]) for i in range(0,len(dataset))]) 
        
        constraints=switchers*((len(measurable)*tolerance_MBDM**2)*np.ones(len(dataset))-residual_term)    
                    
        #print(constraints)
        return constraints
    
    cons=[{'type':'ineq', 'fun':MBDM_constraints}]
    
    def MBDM_objective(optimisation_variables, return_sample_contribution=False):
        
        parameters=optimisation_variables[0:len(theta)]
        switchers=optimisation_variables[len(theta):]
        residuals=np.zeros(shape=(len(dataset),len(sigma)))
        predictions=np.zeros(shape=(len(dataset),len(sigma)))
        measurements=[dataset[i][3] for i in range(0,len(dataset))]
    
        for i in range(0,len(dataset)):
            X=odeint(model,dataset[i][0],[0.0,dataset[i][2]],(dataset[i][1][0:2], parameters,(((simulation_pdc(c_theta,dataset,c_estimate,L,inletpressure_m,params)[1][i]+dataset[i][1][2])/2.0))))
            predictions[i]=X[1]
        
        residuals=((measurements-predictions)/sigma)**2
        residual_term=np.array([np.sum(residuals[i,measurable]) for i in range(0,len(dataset))])
        objective=-np.sum(switchers*((len(measurable)*tolerance_MBDM**2)*np.ones(len(dataset))-residual_term))
    
        if return_sample_contribution:

            return (len(measurable) * tolerance_MBDM ** 2) * np.ones(len(dataset)) - residual_term
    
        return objective

    MBDM_results=minimize(MBDM_objective, data_mining_parameter_set, method='SLSQP', bounds=(optimisation_variables_bounds), options={'disp': True, 'ftol': 1e-20, 'maxiter':300})   
    #MBDM_results=minimize(MBDM_objective, data_mining_parameter_set, method='SLSQP', bounds=(optimisation_variables_bounds), constraints=cons, options={'disp': True, 'ftol': 1e-20})  
    parameters=MBDM_results.x[0:len(theta)]
    switchers=MBDM_results.x[len(theta):]
    
    sample_contribution_to_objective = MBDM_objective(MBDM_results.x, return_sample_contribution=True)
    
    return parameters, switchers, sample_contribution_to_objective

def MBDM_dataset(switchers,dataset,inletpressure_m):
    
    binary_switchers = []
    for i in range(0,len(switchers)):
        if switchers[i]>0.5:
            binary_switcher = 1
        else:
            binary_switcher = 0
        binary_switchers = np.append(binary_switchers,binary_switcher)
    
    datasets = [np.zeros(len(dataset[0]))]
    inletpressures = []
    for i in range(0,len(dataset)):
        if binary_switchers[i] == 1:
            data = dataset[i]
            inletpressure = inletpressure_m[i]
            
            datasets = np.append(datasets,[data],axis=0)
            inletpressures = np.append(inletpressures,[inletpressure],axis=0)
    
    datasets_final = datasets[1:]
    
    return datasets_final, inletpressures

######### Plotting results ##########

def Comp_evolution(estimator,candidate_model,n0,mc,kinetic_parameters):
    
    t1 = np.linspace(0.0,mc,100)

    X1=odeint(candidate_model,n0,t1,args=(np.array([25,350]),kinetic_parameters,1.55,))
   
    plt.plot(t1,X1[:,0],label=r"CH$_4$")
    plt.plot(t1,X1[:,1],label=r"O$_2$")
    plt.plot(t1,X1[:,2],label=r"CO$_2$")
    plt.xlabel('Catalyst mass (g)')
    plt.ylabel('Mole fraction')
    plt.title(estimator,fontdict={'fontsize':13})
    plt.legend()

    plt.show()

def Barplots_1(estimator,residuals,switcher):
    
    len_exp = int(len(residuals)/2)
    
    labels =[]
    if estimator=='MBDM experiment dependent residuals':
        for i in range(0, len_exp):
            if switcher[i]>0:
                label = 'Exp '+str(i+1)
            else:
                label = 'Exp '+str(i+1)+u"$^{\u2020}$"
            labels = np.append(labels,label)
    else:
        labels = ['Exp '+str(i) for i in range(1, len_exp + 1)]

    residuals = np.array(residuals)
    
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1*width, residuals[0:len_exp, 0], width, label=r'CH$_4$')
    rects2 = ax.bar(x, residuals[0:len_exp, 1], width, label=r'O$_2$')
    rects3 = ax.bar(x + 1.0*width, residuals[0:len_exp, 2], width, label=r'CO$_2$')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Residuals')
    ax.set_yscale('log')
    ax.set_title(estimator,fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.26),ncol=3)
    plt.xticks(rotation=90)
    fig.tight_layout()
    
    plt.show()
    
def MBDM_resids(residuals,sample_contribution):
    
    for i in range(0,len(residuals)):
        if sample_contribution[i]<0:
            np.put(residuals[i],[0],0)
            np.put(residuals[i],[1],0)
            np.put(residuals[i],[2],0)
            
    return residuals

def Barplots_2(estimator,residuals,switcher):
    
    len_exp = int(len(residuals)/2)

    labels =[]
    if estimator=='MBDM experiment dependent residuals':
        for i in range(len_exp, len(residuals)):
            if switcher[i]>0:
                label = 'Exp '+str(i+1)
            else:
                label = 'Exp '+str(i+1)+u"$^{\u2020}$"
            labels = np.append(labels,label)
    else:
        labels = ['Exp '+str(i) for i in range(len_exp + 1, len(residuals) + 1)]

    residuals = np.array(residuals)
    
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1*width, residuals[len_exp:len(residuals), 0], width, label=r'CH$_4$')
    rects2 = ax.bar(x, residuals[len_exp:len(residuals), 1], width, label=r'O$_2$')
    rects3 = ax.bar(x + 1.0*width, residuals[len_exp:len(residuals), 2], width, label=r'CO$_2$')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Residuals')
    ax.set_yscale('log')
    ax.set_title(estimator,fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.26),ncol=3)
    plt.xticks(rotation=90)

    fig.tight_layout()

    plt.show()
    
def Switchers_bar_plot_1(switchers):
    
    len_exp = int(len(switchers)/2)

    labels =[]
    for i in range(0, len_exp):
        if switchers[i]>0.5:
            label = 'Exp '+str(i+1)
        else:
            label = 'Exp '+str(i+1)+u"$^{\u2020}$"
        labels = np.append(labels,label)

    switchers = np.array(switchers)
    
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    ax.bar(x, switchers[0:len_exp])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Integer value')
    ax.set_yticks([0,1])
    ax.set_title('Binary switchers',fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xticks(rotation=90)

    fig.tight_layout()

    plt.show()
    
def Switchers_bar_plot_2(switchers):
    
    len_exp = int(len(switchers)/2)
    
    labels =[]
    for i in range(len_exp, len(switchers)):
        if switchers[i]>0.5:
            label = 'Exp '+str(i+1)
        else:
            label = 'Exp '+str(i+1)+u"$^{\u2020}$"
        labels = np.append(labels,label)

    switchers = np.array(switchers)
    
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    ax.bar(x, switchers[len_exp:len(switchers)])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Integer value')
    ax.set_yticks([0,1])
    ax.set_title('Binary switchers',fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xticks(rotation=90)

    fig.tight_layout()

    plt.show()
    
def Parityplot(estimator,measurement,prediction,dataset,tolerance,sigma):
    
    CH4_measurement_array, CH4_prediction_array = [measurement[0][0]], [prediction[0][0]]
    O2_measurement_array, O2_prediction_array = [measurement[0][1]], [prediction[0][1]]
    CO2_measurement_array, CO2_prediction_array = [measurement[0][2]], [prediction[0][2]]
    
    for i in range(1,len(dataset)):
        CH4_measurement, CH4_prediction = [measurement[i][0]], [prediction[i][0]]
        O2_measurement, O2_prediction = [measurement[i][1]], [prediction[i][1]]
        CO2_measurement, CO2_prediction = [measurement[i][2]], [prediction[i][2]]
        CH4_measurement_array, CH4_prediction_array = np.append(CH4_measurement_array,CH4_measurement), np.append(CH4_prediction_array,CH4_prediction)
        O2_measurement_array, O2_prediction_array = np.append(O2_measurement_array,O2_measurement), np.append(O2_prediction_array,O2_prediction)
        CO2_measurement_array, CO2_prediction_array = np.append(CO2_measurement_array,CO2_measurement), np.append(CO2_prediction_array,CO2_prediction)
   
    CH4_array = (CH4_measurement_array,CH4_prediction_array)
    O2_array = (O2_measurement_array,O2_prediction_array)
    CO2_array = (CO2_measurement_array,CO2_prediction_array)
    
    data = (CH4_array, O2_array, CO2_array)
    markers = ("x", "o", "s","^")
    species = ("Methane", "Oxygen", "Carbon dioxide")
    
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for data, marker, group in zip(data, markers, species):
        x, y = data
        ax.scatter(x, y, alpha=0.8, marker=marker, edgecolors='none', s=30, label=group)
    
    
    plt.plot(np.linspace(0.0,0.11,100),np.linspace(tolerance*sigma[1],0.11+tolerance*sigma[1],100),c='gray',linestyle='dotted')
    plt.plot(np.linspace(0.0,0.11,100),np.linspace(0.0,0.11,100),c='gray',linestyle='dashed')
    plt.plot(np.linspace(tolerance*sigma[1],0.11+tolerance*sigma[1],100),np.linspace(0.0,0.11,100),c='gray',linestyle='dotted')
    
    plt.title(estimator,fontdict={'fontsize':13})
    plt.xlim(0, 0.11)
    plt.ylim(0, 0.11)
    plt.xlabel('Measurement (mol/mol)')
    plt.ylabel('Prediction (mol/mol)')
    plt.legend(loc=2)
    plt.show()
    
def Residual_dist(estimator,measurement,prediction,sigma,no_bins):
    
    residuals = []
    for j in range(0,len(measurement)):
        for i in range(0,len(measurement[j])):
            if prediction[j][i]>measurement[j][i]:
                residual =((prediction[j][i]-measurement[j][i])/sigma[i])**2
            else:
                residual =-((prediction[j][i]-measurement[j][i])/sigma[i])**2
            residuals = np.append(residuals,residual)
    
    # the histogram of the data
    n, bins, patches = plt.hist(residuals, no_bins, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(estimator,fontdict={'fontsize':13})
    plt.grid(True)
    plt.show()
            
def Reliability_map(dataset,binary_switchers):

    range_flowrate=[20, 30]
    #range_flowrate=[12.5, 37.5]
    
    range_temperature=[250, 350]
    #range_temperature=[215, 375]

    xx, yy = np.meshgrid(np.linspace(0, 1, 200),
                     np.linspace(0, 1, 200))
    
    X = np.zeros((len(dataset), 2))

    scaler_flowrate = interp1d(range_flowrate, [0, 1])
    scaler_temperature = interp1d(range_temperature, [0, 1])

    inverse_scaler_flowrate = interp1d([0, 1], range_flowrate)
    inverse_scaler_temperature = interp1d([0, 1], range_temperature)

    X[:, 0] = np.array([scaler_flowrate(dataset[i][1][0]) for i in range(len(dataset))])
    X[:, 1] = np.array([scaler_temperature(dataset[i][1][1]) for i in range(len(dataset))])
    
    Y = binary_switchers

    # fit the model
    clf = SVC(kernel='rbf', class_weight='balanced')
    clf.fit(X, Y)
    
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    print(Z)

    levels = [-0.8,-0.4,0.0,0.4,0.8]

    plt.figure()
    im = plt.imshow(Z, interpolation='nearest', extent=(range_flowrate[0], range_flowrate[1], 
                    range_temperature[0], range_temperature[1]), aspect='auto', origin='lower', cmap=plt.cm.gray)
    
    contours_1 = plt.contour(inverse_scaler_flowrate(xx), inverse_scaler_temperature(yy), Z, levels=levels
                           ,linewidths=1.0,linestyles='dashed',colors='gray')
    
    contours_2 = plt.contour(inverse_scaler_flowrate(xx), inverse_scaler_temperature(yy), Z, levels=[0], linewidths=2,
                        colors='black')
    
    cbar = plt.colorbar(im)
                        
    cbar.set_label('Score',labelpad=-40,y=1.06,rotation=0)
    cbar.add_lines(contours_1)
    cbar.add_lines(contours_2)
    
    plt.ylabel('Temperture (Celsius)')
    plt.xlabel('Volumetric flow (ml/min)')
    plt.title('Model reliability map',fontdict={'fontsize':13})
    plt.show() 

def Reliability_map_2(dataset,binary_switchers):

    #range_inlet_conc = [0.01,0.03]
    range_inlet_conc = [0.015,0.025]
    
    range_temperature=[250, 350]
    #range_temperature=[200, 390]

    xx, yy = np.meshgrid(np.linspace(0, 1, 200),
                     np.linspace(0, 1, 200))
    
    X = np.zeros((len(dataset), 2))

    scaler_inlet_conc = interp1d(range_inlet_conc, [0, 1])
    scaler_temperature = interp1d(range_temperature, [0, 1])

    inverse_scaler_inlet_conc = interp1d([0, 1], range_inlet_conc)
    inverse_scaler_temperature = interp1d([0, 1], range_temperature)

    X[:, 0] = np.array([scaler_inlet_conc(dataset[i][0][0]) for i in range(len(dataset))])
    X[:, 1] = np.array([scaler_temperature(dataset[i][1][1]) for i in range(len(dataset))])
    
    Y = binary_switchers

    # fit the model
    clf = SVC(kernel='rbf', class_weight='balanced')
    clf.fit(X, Y)
    
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    print(Z)

    levels = [-0.8,-0.4,0.0,0.4,0.8]

    plt.figure()
    im = plt.imshow(Z, interpolation='nearest', extent=(range_inlet_conc[0], range_inlet_conc[1], 
                    range_temperature[0], range_temperature[1]),aspect='auto',origin='lower', cmap=plt.cm.gray)
    
    contours_1 = plt.contour(inverse_scaler_inlet_conc(xx), inverse_scaler_temperature(yy), Z, levels=levels
                           ,linewidths=1.0,linestyles='dashed',colors='gray')
    
    contours_2 = plt.contour(inverse_scaler_inlet_conc(xx), inverse_scaler_temperature(yy), Z, levels=[0], linewidths=2,
                        colors='black')
    
    cbar = plt.colorbar(im)
                        
    cbar.set_label('Score',labelpad=-40,y=1.06,rotation=0)
    cbar.add_lines(contours_1)
    cbar.add_lines(contours_2)
    
    plt.ylabel('Temperture (Celsius)')
    plt.xlabel(r'Inlet mol fraction of CH$_4$ (mol/mol)')
    plt.xticks(np.linspace(0.015,0.025,6))
    plt.title('Model reliability map',fontdict={'fontsize':13})
    plt.show() 

def Reliability_map_3(dataset,binary_switchers):

    range_inlet_conc = [0.015,0.025]
    #range_inlet_conc = [0.01,0.03]

    range_flowrate = [20, 30]
    #range_flowrate = [17.5, 32.5]

    xx, yy = np.meshgrid(np.linspace(0, 1, 200),
                     np.linspace(0, 1, 200))
    
    X = np.zeros((len(dataset), 2))

    scaler_inlet_conc = interp1d(range_inlet_conc, [0, 1])
    scaler_flowrate = interp1d(range_flowrate, [0, 1])

    inverse_scaler_inlet_conc = interp1d([0, 1], range_inlet_conc)
    inverse_scaler_flowrate = interp1d([0, 1], range_flowrate)

    X[:, 0] = np.array([scaler_inlet_conc(dataset[i][0][0]) for i in range(len(dataset))])
    X[:, 1] = np.array([scaler_flowrate(dataset[i][1][0]) for i in range(len(dataset))])
    
    Y = binary_switchers

    # fit the model
    clf = SVC(kernel='rbf', class_weight='balanced')
    clf.fit(X, Y)
    
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    print(Z)

    levels = [-0.8,-0.4,0.0,0.4,0.8]

    plt.figure()
    im = plt.imshow(Z, interpolation='nearest', extent=(range_inlet_conc[0], range_inlet_conc[1], 
                    range_flowrate[0], range_flowrate[1]),aspect='auto',origin='lower', cmap=plt.cm.gray)
    
    contours_1 = plt.contour(inverse_scaler_inlet_conc(xx), inverse_scaler_flowrate(yy), Z, levels=levels
                           ,linewidths=1.0,linestyles='dashed',colors='gray')
    
    contours_2 = plt.contour(inverse_scaler_inlet_conc(xx), inverse_scaler_flowrate(yy), Z, levels=[0], linewidths=2,
                        colors='black')
    
    cbar = plt.colorbar(im)
                        
    cbar.set_label('Score',labelpad=-40,y=1.06,rotation=0)
    cbar.set_ticks(np.linspace(-1.1,1.5,8))
    cbar.add_lines(contours_1)
    cbar.add_lines(contours_2)
    
    plt.ylabel('Volumetric flow (ml/min)')
    plt.xlabel(r'Inlet mol fraction of CH$_4$ (mol/mol)')
    plt.xticks(np.linspace(0.015,0.025,6))
    plt.title('Model reliability map',fontdict={'fontsize':13})
    plt.show() 

def Binary_switchers_table(dataset,binary_switchers):
    
    experiments = [str(dataset[0][1])]
    for i in range(1,len(dataset)):
        experiment = [str(dataset[i][1])]
        experiments = np.append(experiments,experiment)
    
    experiment_no = list(range(1, len(dataset)+1))
    
    table = pd.DataFrame(np.transpose(np.array([experiments,binary_switchers])),index=experiment_no,columns=[' Experimental conditions ', ' Switchers '])

    print(table)
    
def CH4_Deviations(switcher,prediction,measurement,sigma,tolerance):
    
    experiment_no = []
    CH4_predictions, CH4_measurements, CH4_residuals = [],[],[]
    CH4_contributions = []
    
    for i in range(0,len(switcher)):
        if switcher[i] < 0:
            experiment = i+1
            CH4_prediction, CH4_measurement = prediction[i][0], measurement[i][0]
            CH4_residual = ((CH4_prediction-CH4_measurement)/sigma[0])**2
            CH4_contribution = tolerance**2 - CH4_residual
            
            CH4_predictions, CH4_measurements, CH4_residuals = np.append(CH4_predictions,CH4_prediction), np.append(CH4_measurements,CH4_measurement), np.append(CH4_residuals,CH4_residual)
            CH4_contributions = np.append(CH4_contributions,CH4_contribution)
            experiment_no = np.append(experiment_no,experiment)
            
    table = pd.DataFrame(np.transpose(np.array([CH4_measurements, CH4_predictions, CH4_residuals, CH4_contributions])),
                         index=experiment_no,columns=[' Measurement ', ' Prediction ','Normalised Square Residual','Contribution'])    
    
    print(table)
    
    return CH4_contributions
    
def O2_Deviations(switcher,prediction,measurement,sigma,tolerance):
    
    experiment_no = []
    O2_predictions, O2_measurements, O2_residuals = [],[],[]
    O2_contributions = []
    
    for i in range(0,len(switcher)):
        if switcher[i] < 0:
            experiment = i+1
            O2_prediction, O2_measurement = prediction[i][1], measurement[i][1]
            O2_residual = ((O2_prediction-O2_measurement)/sigma[1])**2
            O2_contribution = tolerance**2 - O2_residual
            
            O2_predictions, O2_measurements, O2_residuals = np.append(O2_predictions,O2_prediction), np.append(O2_measurements,O2_measurement), np.append(O2_residuals,O2_residual)
            O2_contributions = np.append(O2_contributions,O2_contribution)
            experiment_no = np.append(experiment_no,experiment)
            
    table = pd.DataFrame(np.transpose(np.array([O2_measurements, O2_predictions, O2_residuals, O2_contributions])),
                         index=experiment_no,columns=[' Measurement ', ' Prediction ','Normalised Square Residual','Contribution'])    
    
    print(table)
    
    return O2_contributions
    
def CO2_Deviations(switcher,prediction,measurement,sigma,tolerance):
    
    experiment_no = []
    CO2_predictions, CO2_measurements, CO2_residuals = [],[],[]
    CO2_contributions = []
    
    for i in range(0,len(switcher)):
        if switcher[i] < 0:
            experiment = i+1
            CO2_prediction, CO2_measurement = prediction[i][2], measurement[i][2]
            CO2_residual = ((CO2_prediction-CO2_measurement)/sigma[2])**2
            CO2_contribution = tolerance**2 - CO2_residual
            
            CO2_predictions, CO2_measurements, CO2_residuals = np.append(CO2_predictions,CO2_prediction), np.append(CO2_measurements,CO2_measurement), np.append(CO2_residuals,CO2_residual)
            CO2_contributions = np.append(CO2_contributions,CO2_contribution)
            experiment_no = np.append(experiment_no,experiment)
            
    table = pd.DataFrame(np.transpose(np.array([CO2_measurements, CO2_predictions, CO2_residuals, CO2_contributions])),
                         index=experiment_no,columns=[' Measurement ', ' Prediction ','Normalised Square Residual','Contribution'])    
    
    print(table)
    
    return CO2_contributions

def Total_deviation(switcher,CH4_contributions,O2_contributions,CO2_contributions,contribution):
    
    experiment_no = []
    total_contributions = []
    
    for i in range(0,len(switcher)):
        if switcher[i] < 0:
            experiment = i+1
            total_contribution = contribution[i]
            
            experiment_no = np.append(experiment_no,experiment)
            total_contributions = np.append(total_contributions,total_contribution)
            
    table = pd.DataFrame(np.transpose(np.array([CH4_contributions, O2_contributions, CO2_contributions, total_contributions])),
                         index=experiment_no,columns=[r' CH$_4$ ',r' O$_2$ ',r' CO$_2$ ',' Total ']) 
    
    print(table)
    
    return total_contributions, experiment_no

def heatmap_data(experiment,CH4_contribution,O2_contribution,CO2_contribution,total_contribution):
    
    CH4_cont = np.around(CH4_contribution[experiment],decimals=1)
    O2_cont = np.around(O2_contribution[experiment],decimals=1)
    CO2_cont = np.around(CO2_contribution[experiment],decimals=1)
    total_cont = np.around(total_contribution[experiment],decimals=1)
    
    return [CH4_cont,O2_cont,CO2_cont,total_cont]
    
def Heatmap(CH4_contribution,O2_contribution,CO2_contribution,total_contribution,experiment_no):
    
    Exp_no = experiment_no.astype(int)
    
    labels = ['Exp '+str(Exp_no[i]) for i in range(0, len(Exp_no))]
    
    headings = [r' CH$_4$ ',r' O$_2$ ',r' CO$_2$ ',' Total ']
    
    dataset = np.array([heatmap_data(0,CH4_contribution,O2_contribution,CO2_contribution,total_contribution)])
    
    for i in range(1,len(total_contribution)):
        dataset = np.append(dataset,[heatmap_data(i,CH4_contribution,O2_contribution,CO2_contribution,total_contribution)],axis=0)
        
    ax = sns.heatmap(dataset, annot=True, cbar_kws={'label': 'Score'})
    
    plt.xlabel('Species') 
    ax.set_title('Sample contributions',fontsize=13)
    ax.set_xticklabels(headings)
    ax.set_yticklabels(labels,rotation=360)
    
    plt.show()
    
###################### Statistical tests ###################

########### Chi-square test ##########

def chisquare_test(alpha,n_prelim,n_y,n_theta,Phi):
    
    conf_level = 1.0 - alpha
    dof = n_prelim * n_y - n_theta
    
    ref_chisquare = st.chi2.ppf((conf_level),dof)
    chisquare_value = Phi
    p_value = 1 - st.chi2.cdf(chisquare_value, dof)
    
    return [ref_chisquare, chisquare_value]
    
########### t-test #############

### Generation of perturbation matrix ###

def perturbation(epsilon,parameter_estimate,n_theta):      
    perturbated_matrix = np.zeros([n_theta+1,n_theta])
    for j in range(n_theta):
        for k in range(n_theta):
            if j==k:
                perturbated_matrix[j,k] = parameter_estimate[j] * (1 + epsilon)
            else:
                perturbated_matrix[j,k] = parameter_estimate[k]
    for j in range(n_theta):
        perturbated_matrix[-1,j] = parameter_estimate[j]
    return perturbated_matrix
    
### Generation of sensitivity matrix ###

def sensitivity(epsilon,parameter_estimate,n_theta,n_y,candidate_model,u,theta,x0,t,Pavg):
    solutions_sen = []
    for theta in perturbation(epsilon,parameter_estimate,n_theta):
        soln_sen = odeint(candidate_model, x0, t, args=(u,theta,Pavg))
        solutions_sen.append(soln_sen)
    yhat_sen_array = np.zeros([n_theta+1,n_y])
    for i in range(n_theta+1):
        yhat_sen_array[i] = np.asarray(solutions_sen)[i][-1][0:n_y]
    sensitivity_matrix = np.zeros([len(parameter_estimate),n_y])
    for j in range(len(parameter_estimate)):
        for k in range(n_y):
            sensitivity_matrix[j,k] = ((yhat_sen_array[j,k] - yhat_sen_array[-1,k])/(epsilon*parameter_estimate[j]))
    return sensitivity_matrix
            
def obssensitivity(epsilon,parameter_estimate,n_theta,n_y,n_prelim,candidate_model,theta,t,dataset,mc,c_theta,c_estimate,L,inletpressure_m,params):
    sensitivity_matrices = list(range(n_prelim))
    for j in range(n_prelim):
        sensitivity_matrices[j] = sensitivity(epsilon,parameter_estimate,n_theta,n_y,candidate_model,[dataset[j][1][0],dataset[j][1][1]],theta,dataset[j][0],np.linspace(0.0,mc,5),(((simulation_pdc(c_theta,dataset,c_estimate,L,inletpressure_m,params)[1][j]+dataset[j][1][2])/2.0)))
    obs_sensitivity_matrices = np.asarray(sensitivity_matrices)
    return obs_sensitivity_matrices,obs_sensitivity_matrices.shape

### Fisher Information function ###
    
def Fisherinformation(epsilon,parameter_estimate,n_theta,n_y,candidate_model,u,theta,x0,t,sigma,Pavg):
    Fisher=np.zeros([n_theta,n_theta])
    for k in range(n_y):
        Fisher = Fisher + (1/(sigma[k]**2)) * np.outer(sensitivity(epsilon,parameter_estimate,n_theta,n_y,candidate_model,u,theta,x0,t,Pavg)[:,k],sensitivity(epsilon,parameter_estimate,n_theta,n_y,candidate_model,u,theta,x0,t,Pavg)[:,k])
    return Fisher
        
### Evaluation of FIM from performed experiments ###
        
def obs_Fisher(epsilon,parameter_estimate,n_theta,n_y,n_prelim,candidate_model,theta,x0,t,sigma,dataset,mc,c_theta,c_estimate,L,inletpressure_m,params):
    obs_Fisherinformation = list(range(n_prelim))
    for j in range(n_prelim):
        obs_Fisherinformation[j] = Fisherinformation(epsilon,parameter_estimate,n_theta,n_y,candidate_model,[dataset[j][1][0],dataset[j][1][1]],theta,dataset[j][0],t,sigma,(((simulation_pdc(c_theta,dataset,c_estimate,L,inletpressure_m,params)[1][j]+dataset[j][1][2])/2.0)))
    obs_Fisherinformation_array = np.asarray(obs_Fisherinformation)
    global_obs_Fisher = sum(obs_Fisherinformation_array[j] for j in range(n_prelim))
    return global_obs_Fisher
            
### Evaluation of covariance matrix from performed experiments ###
    
def obs_covariance(epsilon,parameter_estimate,n_theta,n_y,n_prelim,candidate_model,theta,x0,t,sigma,dataset,mc,c_theta,c_estimate,L,inletpressure_m,params):
    obs_variance_matrix = np.linalg.inv(obs_Fisher(epsilon,parameter_estimate,n_theta,n_y,n_prelim,candidate_model,theta,x0,t,sigma,dataset,mc,c_theta,c_estimate,L,inletpressure_m,params))
    return obs_variance_matrix

### Evaluation of correlation matrix from the performed experiments ### 
        
def correlation(n_theta,obscovariancematrix):
    correlationmatrix = np.zeros([n_theta,n_theta])
    for i in range(n_theta):
        for j in range(n_theta):
            correlationmatrix[i,j] = obscovariancematrix[i,j]/(np.sqrt(obscovariancematrix[i,i] * obscovariancematrix[j,j]))
    return correlationmatrix
    
### statistical infernce testing for assessing quality of parameter estimates (Students t-test) ###
    
def t_analysis(n_theta,obscovariancematrix,alpha,n_prelim,n_y,parameter_estimate):  
    
    dof = n_prelim * n_y - n_theta
    conf_level = 1.0 - alpha
    variances = np.zeros(n_theta)
    t_values = np.zeros(n_theta)
    conf_interval = np.zeros(n_theta)
    for j in range(n_theta):
        conf_interval[j] = np.sqrt(obscovariancematrix[j,j]) * st.t.ppf((1 - (alpha/2)), dof)
        t_values[j] = parameter_estimate[j]/(conf_interval[j])
    t_ref = st.t.ppf((1-alpha),dof)
    return conf_interval,t_values,t_ref