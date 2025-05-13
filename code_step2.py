import numpy as np
import cvxpy as cp

def generate_profile(num_minutes=60, lower_bound=220, upper_bound=600, max_step=35, seed=42):
    """
    Generate a single consumption load profile over a given number of minutes.
    Ensures that:
      - Consumption at each minute is within [lower_bound, upper_bound]
      - Change between consecutive minutes does not exceed max_step (kW)
    """
    np.random.seed(None)

    profile = np.zeros(num_minutes)
    # Initialize the first minute consumption uniformly within [220, 600]
    profile[0] = np.random.uniform(lower_bound, upper_bound)
    
    for t in range(1, num_minutes):
        # Calculate allowable step size so that the next value remains in [220, 600]
        current_value = profile[t - 1]
        min_step = max(-max_step, lower_bound - current_value)
        max_step_allowed = min(max_step, upper_bound - current_value)
        # Draw a random step from the allowable range uniformly
        step = np.random.uniform(min_step, max_step_allowed)
        profile[t] = current_value + step
        
    return profile

def cvar(profiles, eps=0.1, verbose=False):
    num_profiles, num_minutes = profiles.shape

    # Problem variables
    offer_capacity = cp.Variable(nonneg=True)
    beta = cp.Variable()
    zeta = cp.Variable((num_minutes, num_profiles))

    # Problem constraints
    constraints=[]
    constraints+=[beta<=0]
    for i in range(num_minutes):
        for j in range(num_profiles):
            constraints+=[
                offer_capacity - profiles[j,i] <= zeta[i,j]
            ]
            constraints+=[
                beta<=zeta[i,j]
            ]
    constraints+=[
        (1/(num_profiles*num_minutes))*cp.sum(zeta) <= (1-eps)*beta
    ]

    # Solve problem
    problem = cp.Problem(cp.Maximize(offer_capacity), constraints)
    problem.solve(verbose=verbose)

    # Return results
    return offer_capacity.value, problem

def also_x(profiles, eps=0.1, tol=10e-5, M=1e4, verbose=False):
    num_profiles, num_minutes = profiles.shape
    q_underbar=0

    # Todo: not sure if this is correct
    q_overbar=tol*num_profiles**2
    # Alternative
    #q_overbar=tol*num_profiles*num_minutes
    it_counter=0
    while (q_overbar-q_underbar)>=tol:
        q = (q_underbar+q_overbar)/2
        offer_capacity=cp.Variable(nonneg=True)
        y = cp.Variable((num_minutes,num_profiles))
        constraints=[]
        constraints+=[
            y>=0, y<=1
        ]
        for i in range(num_minutes):
            for j in range(num_profiles):
                constraints+=[
                    offer_capacity - profiles[j,i] <= M*y[i,j]
                ]
        constraints+=[
            cp.sum(y)<=q
        ]
        problem = cp.Problem(cp.Maximize(offer_capacity), constraints)
        problem.solve(verbose=verbose)
        
        prob = (1/num_profiles*num_minutes)*np.sum(y.value)
        if(1-prob>=1-eps):
            q_underbar=q
        else:
            q_overbar=q

        print(f'Iteration: {it_counter}')
        print(prob, q_overbar,q_underbar, q_overbar-q_underbar)
        it_counter+=1

    return offer_capacity.value, problem
