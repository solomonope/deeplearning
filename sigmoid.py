import math as Math;

#perceptron
#sigmoid
#tahn
#ReLu --> rectified linear unit
def sigmoid(x):
    return  1/ (1 + Math.e ** -x);



if __name__ == "__main__":
    print(sigmoid(-100))
    print(sigmoid(100))
    print(sigmoid(-1))
    print(sigmoid(2))