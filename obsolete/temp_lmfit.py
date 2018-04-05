from numpy import exp, sin
import numpy as np
from lmfit import minimize, Parameters


def residual(params, x, data, eps_data):
    amp = params['amp']
    phaseshift = params['phase']
    freq = params['frequency']
    decay = params['decay']

    model = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)

    return (data-model) / eps_data


x = np.linspace(0.1, 5, 200)
data = 10 * sin(x*3 + .2) * exp(-x*x*.007)
eps_data = np.ones_like(data)
params = Parameters()

params.add('amp', value=10, min=10, max=11, vary=True, brute_step=2)
params.add('decay', value=0.007, min=0.006, max=0.007, vary=True, brute_step=0.001)
params.add('phase', value=0.2, min=0.1, max=0.3, vary=False)
params.add('frequency', value=3.0, min=1, max=2, vary=True, brute_step=1)

out = minimize(residual, params, args=(x, data, eps_data), method="brute")

print(out)

names = out.var_names
index = names.index("frequency")

print(index, "index of freq param")
brute = out.brute_grid
print(brute.shape)
print("done")