import numpy as np
import matplotlib.pyplot as plt
 
# Use matplotlib's built-in math text renderer
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
 
# Define the function
def f(K,gamma,x):
    return K*np.cos(x)/(1 + np.exp(-gamma*np.cos(x)))

def f(x, k):
    r"""
    Weighting function for smooth transition between two states.

    Parameters:
    - x: The input value.
    - k: The gain for steepness of the transition.

    Returns:
    - The weighted value.
    """

    return 1 / (1 + np.exp(-k * (x - 0.5)))
 
# # Define the parameters
K = 5
 
# Generate x values
x = np.linspace(-2/3, 1+2/3, 100)
 
 
# Generate y values

k = [5, 7, 10]

y5 = f(x, 5)
y8 = f(x, 7)
y10 = f(x, 10)

 
# Create the plot
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x, f(x,k[0]), 'k',label=f'$k = {k[0]}$')
ax.plot(x, f(x,k[1]), 'm',label=f'$k = {k[1]}$')
ax.plot(x, f(x,k[2]), 'c',label=f'$k = {k[2]}$')
plt.axvline(x=0, color='r', linestyle='--', label=r'$q_i$ at soft limit start')
plt.axvline(x=1, color='b', linestyle='--', label=r'$q_i$ at soft limit end')
plt.axvline(x=1+2/3, color='k', linestyle='--', label=r'$q_i$ at hard limit') 
plt.ylim(-0, 1)

 
ax.legend()
plt.xlabel(r'Distance to limit $x$')
ax.set_ylabel(r'Weight $\lambda$', rotation='vertical', loc='center', labelpad=1)
 
# Show the plot
plt.show()