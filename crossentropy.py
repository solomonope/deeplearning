import numpy as NP;

"""
cost functions

Quadratic --> SSE, residual sum of squares

cross entropy
"""
def cross_entropy(y, a):
    return -1 * (y * NP.log(a) + (1 - y) * NP.log(1 - a));



if __name__ == "__main__":
    print(cross_entropy(0, 0.01))
    print(cross_entropy(1, 0.99))
    print(cross_entropy(0, 0.3))
