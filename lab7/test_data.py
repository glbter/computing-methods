import numpy as np

initial = [0.7, 0.8]
init = [0.65, 0.85]

system = [
    lambda x1, x2: 2 * (x1**2) - x1 + x2**2 - 1,
    lambda x1, x2: x2 - np.tan(x1)
]

system_special = [
    lambda x1, x2: 2 * (x1**2) + x2**2 - 1,
    lambda x1, x2: -np.tan(x1)
]

system_super_special = [
    lambda x1, x2: -np.arctan(x2),
    lambda x1, x2: np.sqrt(2*x1**2 - x1 - 1)
]

system_super_special2 = [
    lambda x1, x2: -np.arctan(x2),
    lambda x1, x2: -np.sqrt(2*x1**2 - x1 - 1)
]

special_jacobian = [
    [lambda x1, x2: 4*x1, lambda x1, x2: 2*x2],
    [lambda x1, x2: -1/(np.cos(x1)**2), lambda x1, x2: 0]
]

jacobian = [
    [lambda x1, x2: 4*x1 - 1, lambda x1, x2: 2*x2],
    [lambda x1, x2: -1/(np.cos(x1)**2), lambda x1, x2: 1]
]

test_system = [
    lambda x1, x2: 0.1 * x1**2 + x1 + 0.2 * x2**2 - 0.3,
    lambda x1, x2: 0.1 * x1**2 + x2 - 0.1 * x1 * x2 - 0.7
]

def f1dx1_test(x1, x2):
    return 0.2 * x1 + 1

test_jacobian = [
    [f1dx1_test, lambda x1, x2: 0.4 * x2],
    [lambda x1, x2: 0.2 * x1 - 0.1 * x2, lambda x1, x2: 1 - 0.1 * x1]
]

test_special_jacobian = [
    [lambda x1, x2: -0.2*x1, lambda x1, x2: -0.4*x2],
    [lambda x1, x2: -0.4*x1 + 0.1*x2, lambda x1, x2: 0.1*x1]
]

test_special = [
    lambda x1, x2: 0.3 - 0.1*x1**2 - 0.2*x2**2,
    lambda x1, x2: 0.7 - 0.2*x1**2 + 0.1*x1*x2
]

test2_system = [
        lambda x, y: np.sin(2*x - y) - 1.2*x - 0.4,
        lambda x, y: 0.8* x**2 + 1.5* y**2 - 1
    ]

test2_jacobian = [
    [lambda x, y: 2*np.cos(2*x - y) - 1.2, lambda x, y: np.cos(2*x - y)],
    [lambda x, y: 1.6*x, lambda x, y: 3*y]
]
