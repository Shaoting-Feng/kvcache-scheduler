import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

def convex_optimize(doc_info, S):
    threshold = 0.5

    doc_info = doc_info.copy()

    # Step 1: Define the fitting points for the quality curve
    compression_rates = np.array([0, 0.499, 0.780, 1])
    qualities = np.array([0, 0.85, 0.97, 1])

    # Fit a quadratic function Q(r)
    coefficients = np.polyfit(compression_rates, qualities, 2)
    c2, c1, c0 = coefficients  # Extract coefficients of the quadratic function

    # Filter documents that have not been sent
    doc_info = doc_info[doc_info['sent'] == False]

    # Prepare variables for optimization
    n_docs = len(doc_info)
    v3_sizes = doc_info['v3_size'].values
    probs = doc_info['prob'].values

    # Define the optimization variable for compression rates
    r = cp.Variable(n_docs)
    r_binary = cp.Variable(n_docs, boolean=True)

    # Define the objective function and constraints
    objective = cp.Maximize(cp.sum(cp.multiply(probs, (c2 * r**2 + c1 * r + c0))))
    constraints = [
        cp.sum(cp.multiply(r, v3_sizes)) <= S,  # Total compressed size must not exceed S
        r >= 0,  # Compression rate must be at least 0
        r <= 0.778,  # Compression rate must not exceed 1
        r_binary * threshold <= r,  
        r_binary >= r
    ]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    cplex_params = {
        "mip.tolerances.mipgap": 0.04  # 设置 MIP 问题的容忍度
    }
    problem.solve(solver=cp.CPLEX, cplex_params=cplex_params)

    print(r.value.round(2))
    doc_info['optimal_compression_rate'] = r.value.round(2)
    doc_info = doc_info[doc_info['optimal_compression_rate'] != 0]
    doc_info = doc_info.sort_values(by='prob', ascending=False)

    # Create output list of tuples (doc_id, optimal_compression_rate)
    results = list(zip(doc_info['doc_id'], doc_info['optimal_compression_rate']))

    return results

# # Example usage
# csv_file_path = '/Users/sfeng/Documents/CMSC331_333/flowsim/trace/doc_stats.csv'
# doc_info = pd.read_csv(csv_file_path)
# doc_info['sent'] = False  # Assume all documents are not sent yet
# doc_info['prob'] = 1
# S = 2812.5  # Assume a total size limit for compression
# optimized_results = convex_optimize(doc_info, S)
# print(optimized_results)