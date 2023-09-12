import numpy as np
from sympy import isprime
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations_with_replacement
from functools import reduce
import operator
from concurrent.futures import ProcessPoolExecutor, as_completed

def is_below_energy_limit(cap, num):
    if isprime(num) and num**2 <= cap:
        return True
    elif not isprime(num) and num < cap:
        return True
    else:
        # print(f'{num} is not below the energy limit')
        pass

def generate_kets(cap):
    hilbert_space = []
    for i in tqdm(range(1, cap)):
        if is_below_energy_limit(cap, i):
            hilbert_space.append(i)
    return np.array(hilbert_space)

def partition_hilbert_space(arr):
    primes = []
    notPrimes = []
    for i in range(1, len(arr)):
        if isprime(arr[i]):
            primes.append(float(arr[i]))
        else:
            notPrimes.append(float(arr[i]))
    return np.array(primes), np.array(notPrimes)

# Define truncated prime zeta function (sum over primes)
def truncated_sum(exp, partition, primesum=True):
    if primesum:
        return np.sum(partition[0]**-exp)
    else:
        return np.sum(partition[1]**-exp)

# Define the two point correlation function for X = sum_{p \in P}delta_p
def two_pt_correlation_delta_observable(beta, t, scale, partition):
    # 2pt correlation
    numerator = truncated_sum(2*beta - scale*1j*t, partition, primesum=True) + truncated_sum(beta, partition, primesum=False)
    denominator = truncated_sum(2*beta, partition, primesum=True) + truncated_sum(beta, partition, primesum=False)
    result = (numerator/denominator)*(truncated_sum(1j*t, partition, primesum=True)/(partition[0].shape[0]))

    return result

def compute_correlations_multiprocessing(i, j, beta_values, t_values, scale, partition):
    return two_pt_correlation_delta_observable(beta_values[i], t_values[j], scale, partition)

def main():
    global beta_values
    global t_values
    global scale

    cap = int(input("Set an energy cap: "))

    #Scale the time oscillations (keep it 1 to be faithful)
    scale = int(input("Set scale: "))

    # Define the number of points plotted (i.e. resolution)
    num_plot_plots = int(input("Set resolution scale: "))

    # Define fixed beta for 2D plot
    epsilon = 0.05
    beta_of_interest = 0.5
    # fixed_betas = [beta_of_interest - 2*epsilon, beta_of_interest - epsilon, beta_of_interest, beta_of_interest + epsilon, beta_of_interest + 2*epsilon]
    # fixed_betas = [beta_of_interest - epsilon, beta_of_interest, beta_of_interest + epsilon]
    # beta_values_to_plot = [i for i in fixed_betas]

    # If you want to manually enter betas
    beta_values_to_plot = [0.01, 1, 2.75]

    # Define domains for 3D plot
    beta_min = 0
    beta_max = 3
    time_min = 0
    time_max = float(input("Set time_max: "))

    hilbert_space = generate_kets(cap)

    partitioned_space = partition_hilbert_space(hilbert_space)
    
    print(partitioned_space[0])
    print(partitioned_space[1])

    # Generate data points for beta and t
    beta_values = np.linspace(beta_min, beta_max, num_plot_plots)
    t_values = np.linspace(time_min, time_max, num_plot_plots)

    # Initialize meshgrid for beta and t
    B, T = np.meshgrid(beta_values, t_values)

    # Calculate the correlation values
    Z = np.zeros_like(B, dtype=complex)

    futures = {}
    with ProcessPoolExecutor() as executor:
        for i in tqdm(range(len(beta_values))):
            for j in range(len(t_values)):
                future = executor.submit(compute_correlations_multiprocessing, i, j, beta_values, t_values, scale, partitioned_space)
                futures[future] = (i, j)
        print("Processes initialized")

        # Create a tqdm object
        pbar = tqdm(total=num_plot_plots**2)
        
        for future in as_completed(futures):
            i, j = futures[future]
            try:
                Z[j, i] = future.result()
                pbar.update(1)
            except Exception as e:
                print(f"Failed to get result for i={i}, j={j}. Error: {e}")
                pbar.update(1)

    # Create a figure
    fig = plt.figure(num=f'energy_limit_{cap}_resolution_{num_plot_plots}_time_scale_{scale}', figsize=(10, 8))

    # Add 3D subplot
    ax3d = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1, projection='3d')

    # Set orientation 3D subplot
    ax3d.view_init(30, 60)

    # Plot the 3D surface
    surf = ax3d.plot_surface(B, T, np.abs(Z), cmap='rainbow', vmin=0, vmax=0.5)

    # Add 3D axis labels and title
    ax3d.set_xlabel(r'$\beta$ (inverse temperature)')
    ax3d.set_ylabel(f'Time (seconds x{scale})')
    ax3d.set_zlabel('correlation_value')

    # Add color bar
    fig.colorbar(surf)

    # Loop to create the 2D subplots
    subplot_positions = [(0, 1), (1, 0), (1, 1)]
    for i, fixed_beta in enumerate(beta_values_to_plot):
        ax2d = plt.subplot2grid((2, 2), subplot_positions[i], colspan=1, rowspan=1)
        Z_fixed_beta = np.array([two_pt_correlation_delta_observable(fixed_beta, t, scale, partitioned_space) for t in t_values])
        ax2d.plot(t_values, np.abs(Z_fixed_beta))
        ax2d.set_title(f'Fixed beta = {fixed_beta}')
        ax2d.set_xlabel(f'Time (seconds x{scale})')
        ax2d.set_ylabel('correlation_value')
    # Set default save format to PDF
    plt.rcParams['savefig.format'] = 'pdf'

    # Set default save path to custom
    save_directory = "./prime_interaction_plots"
    plt.rcParams["savefig.directory"] = save_directory

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()