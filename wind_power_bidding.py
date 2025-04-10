import cvxpy as cp
import numpy as np

def wind_pp_bidding(wind_forecasts, spot_price_forecasts, imbalance_forecasts, n_scenarios, price_scheme='one_price'):

    """
    wind_forecasts: numpy array of wind power forecasts
    spot_price_forecasts: numpy array of spot price forecasts
    imbalance_forecasts: numpy array of imbalance forecasts
    n_scenarios: number of scenarios for the optimization (list with number of wf scenarios, number of spot prices scenarios and number of imbalance scenarios)
    price_scheme: 'one_price' or 'two_price'
    """

    # Input information
    n1,n2,n3 = n_scenarios
    T = np.shape(wind_forecasts)[0]  # number of time steps
    balance_price_down = np.zeros((T,n1,n3))
    balance_price_up = np.zeros((T,n1,n3))
    f_down = 1.6; f_up = 0.5
    for t in range(T):
        for s1 in range(n1):
            for s3 in range(n3):
                if price_scheme == 'two_price': 
                    if imbalance_forecasts[t,s3] > 0:  # System is long
                        balance_price_up[t,s1,s3] = spot_price_forecasts[t,s1]
                        balance_price_down[t,s1,s3] = f_down * spot_price_forecasts[t,s1]
                    else:
                        balance_price_up[t,s1,s3] = f_up * spot_price_forecasts[t,s1]
                        balance_price_down[t,s1,s3] = spot_price_forecasts[t,s1]
                else: 
                    balance_price_up[t,s1,s3] = f_up * spot_price_forecasts[t,s1]
                    balance_price_down[t,s1,s3] = f_down * spot_price_forecasts[t,s1]

    # Define the optimization variables
    wind_power_bid = cp.Variable(T) # wind power bid (in per unit of nominal capacity)
    imbalance_up = cp.Variable((T,n2)) # imbalance up
    imbalance_down = cp.Variable((T,n2)) # imbalance down
    imbalance = cp.Variable((T,n2)) # imbalance

    # Define the constraints
    constraints = []
    constraints += [wind_power_bid >= 0]  # wind power bid must be non-negative
    constraints += [wind_power_bid <= 5]  # wind power bid must max
    constraints += [imbalance_up >= 0]  # imbalance up must be non-negative
    constraints += [imbalance_down >= 0]  # imbalance down must be non-negative
    constraints += [imbalance == imbalance_up - imbalance_down]  # imbalance is the difference between imbalance up and down
    for t in range(T):
        for s2 in range(n2):
            constraints += [imbalance[t,s2] == wind_forecasts[t,s2] - wind_power_bid[t]]  # imbalance is the difference between wind power bid and wind forecast

    # Define the objective function
    objective = 0  
    for s1 in range(n1):
        for s2 in range(n2):
            for s3 in range(n3):
                p1 = cp.sum(cp.multiply(wind_power_bid, spot_price_forecasts[:,s1]))
                p2=0
                for t in range(T):
                    # System is long (too much production) --> pay people at imbalance down price
                    p2 += imbalance_up[t,s2] * balance_price_up[t,s1,s3] # penalised for overproduction
                    # System is short (too little production) --> pay people at imbalance price up
                    p2 -= imbalance_down[t,s2] * balance_price_down[t,s1,s3] # penalised for underproduction
                objective += 1/(n1*n2*n3) * (p1+p2)


    # Define the optimization problem
    problem = cp.Problem(cp.Maximize(objective), constraints)
    # Solve the problem
    problem.solve()
                
    return wind_power_bid.value, problem