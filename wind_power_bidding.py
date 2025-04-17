import cvxpy as cp
import numpy as np

def wind_pp_bidding(wind_forecasts, spot_price_forecasts, imbalance_forecasts, n_scenarios, f_up, f_down, price_scheme='one_price'):

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

    # Define the optimization variables
    wind_power_bid = cp.Variable(T) # wind power bid (in per unit of nominal capacity)
    imbalance_up = cp.Variable((T,n2)) # imbalance up
    imbalance_down = cp.Variable((T,n2)) # imbalance down
    imbalance = cp.Variable((T,n2)) # imbalance

    # Define the constraints
    constraints = []
    constraints += [wind_power_bid >= 0]  # wind power bid must be non-negative
    constraints += [wind_power_bid <= 1]  # wind power bid must max
    constraints += [imbalance_up >= 0]  # imbalance up must be non-negative
    constraints += [imbalance_up <= 1]  # imbalance up must max rated power
    constraints += [imbalance_down >= 0]  # imbalance down must be non-negative
    constraints += [imbalance_down <= 1]  # imbalance down must max rated power
    
  
    for s2 in range(n2):
        constraints += [imbalance[:,s2] == wind_forecasts[:,s2] - wind_power_bid[:]]  # imbalance is the difference between wind power bid and wind forecast
        constraints += [imbalance[:,s2] == imbalance_up[:,s2] - imbalance_down[:,s2]]  # imbalance is the difference between imbalance up and down

    # Define the objective function
    objective = 0  
    for s1 in range(n1):
        for s2 in range(n2):
            for s3 in range(n3):
                p1 = cp.sum(cp.multiply(wind_power_bid, spot_price_forecasts[:,s1]))
                p2=cp.sum(0)
                for t in range(T):
                    
                    if price_scheme == 'one_price':
                        # According to example slides --> fixed price for wind pp independent of system state
                        """
                        p2 += imbalance_up[t,s2] * f_down * spot_price_forecasts[t,s1] # paid for overproduction at regulation down price
                        p2 -= imbalance_down[t,s2] * f_up * spot_price_forecasts[t,s1] # pay back for underproduction at regulation up price
                        """
                        # In this case --> incentivised to bid what you expect to produce
                        # What I would expect intuitively:
                        if imbalance_forecasts[t,s3] >= 0: # System is long
                            p2 += imbalance_up[t,s2] * f_down * spot_price_forecasts[t,s1] # paid for overproduction at regulation down price --> less than could have earned in DA
                            p2 -= imbalance_down[t,s2] * f_down * spot_price_forecasts[t,s1] # pay back for underproduction at regulation up price (making profit with DA as imbalance is desired)
                        else: # System is short
                            p2 += imbalance_up[t,s2] * f_up * spot_price_forecasts[t,s1] # paid for overproduction at regulation down price (making profit with DA as imbalance is desired)
                            p2 -= imbalance_down[t,s2] * f_up * spot_price_forecasts[t,s1] # pay back for underproduction at regulation up price, paying more than in DA

                    elif price_scheme == 'two_price':
                        if imbalance_forecasts[t,s3] >= 0: # System is long
                            p2 += imbalance_up[t,s2] * f_down * spot_price_forecasts[t,s1] # paid for overproduction at regulation down price --> less than could have earned in DA
                            p2 -= imbalance_down[t,s2] * spot_price_forecasts[t,s1] # pay back for underproduction at DA price --> removing incentive to be wrong
                        else: # System is short
                            p2 += imbalance_up[t,s2] * spot_price_forecasts[t,s1] # paid for overproduction at DA price --> removing incentive to be wrong
                            p2 -= imbalance_down[t,s2] * f_up * spot_price_forecasts[t,s1] # pay back for underproduction at regulation up price, paying more than in DA
                objective += 1/(n1*n2*n3) * (p1+p2)

    obj = cp.Maximize(objective)
    # Define the optimization problem
    problem = cp.Problem(obj, constraints)
    # Solve the problem
    problem.solve()
    
    # Print the objective value
    print("Objective value: ", problem.value)
    return wind_power_bid.value, problem,  imbalance.value, imbalance_up.value, imbalance_down.value


def compute_profit(wind_power_bid, wind_realization, spot_price, imbalance, f_down, f_up, price_scheme='one_price'):
    """
    wind_power_bid: numpy array of wind power bids
    wind_realization: numpy array of wind power realizations
    spot_price: numpy array of spot prices
    imbalance: numpy array of imbalances
    price_scheme: 'one_price' or 'two_price'
    """
    
    T = np.shape(wind_power_bid)[0]  # number of time steps
    profit = 0

    for t in range(T):
        p1 = wind_power_bid[t] * spot_price[t]
        p2 = 0
        if price_scheme == 'one_price':
            if imbalance[t] >= 0: # System is long
                p2 += (wind_realization[t]-wind_power_bid[t]) * f_down * spot_price[t] # paid for overproduction at regulation down price --> less than could have earned in DA
            else: # System is short
                p2 += (wind_realization[t]-wind_power_bid[t]) * f_up * spot_price[t] # paid for overproduction at regulation up price --> more than could have earned in DA

        elif price_scheme == 'two_price':
            if imbalance[t] >= 0: # System is long
                p2 += max((wind_realization[t]-wind_power_bid[t]),0) * f_down * spot_price[t] # paid for overproduction at regulation down price --> less than could have earned in DA
                p2 += min((wind_realization[t]-wind_power_bid[t]),0) * spot_price[t] # pay back for underproduction at DA price --> removing incentive to be wrong
            else: # System is short
                p2 += max((wind_realization[t]-wind_power_bid[t]),0) * spot_price[t] # paid for overproduction at DA price --> removing incentive to be wrong
                p2 += min((wind_realization[t]-wind_power_bid[t]),0) * f_up * spot_price[t] # pay back for underproduction at regulation up price, paying more than in DA
        profit += (p1 + p2)

    return profit