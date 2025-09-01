import numpy as np 
def step(x): 
    return 1 if x >= 0 else 0 
def perceptron(x1, x2, weights, bias): 
    x = x1 * weights[0] + x2 * weights[1] + bias 
    return step(x) 
 
# ----- AND Function ----- 
print("AND Function:") 
weights_and = [1, 1] 
bias_and = -1.5  # Only (1,1) gives 1 
for x1 in [0, 1]: 
    for x2 in [0, 1]: 
        output = perceptron(x1, x2, weights_and, bias_and) 
        print(f"{x1} AND {x2} = {output}") 
 
# ----- OR Function ----- 
print("\nOR Function:") 
weights_or = [1, 1] 
bias_or = -0.5  # Any one input as 1 gives output 1 
for x1 in [0, 1]: 
    for x2 in [0, 1]: 
        output = perceptron(x1, x2, weights_or, bias_or) 
        print(f"{x1} OR {x2} = {output}")