import sys
import numpy as np
from scipy.special import softmax

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def main():
    train_x = sys.argv[1]
    train_y = sys.argv[2]
    test_x = sys.argv[3]
    array_trainx = np.loadtxt(train_x)
    array_trainy = np.loadtxt(train_y)
    # normalize train_x
    colnumx = array_trainx.shape[1]
    rownumx = array_trainx.shape[0]

    # XminCol=np.zeros((1,colnumx))
    # for i in range(rownumx):
    #     for j in range(colnumx):
    #         if array_trainx[i][j] < XminCol[0][j]:
    #             XminCol[0][j] = array_trainx[i][j]
    # XmaxCol=np.zeros((1,colnumx))
    # for i in range(rownumx):
    #     for j in range(colnumx):
    #         if array_trainx[i][j] > XmaxCol[0][j]:
    #             XmaxCol[0][j] = array_trainx[i][j]

    for i in range(rownumx):
        for j in range(colnumx):
            array_trainx[i][j] = (array_trainx[i][j] / 255)

    # Initialize random parameters and inputs
    temp = 15
    W1 = np.random.rand(temp, 784)  # shape 15,784
    b1 = np.random.rand(temp)  # shape 15
    W2 = np.random.rand(10, temp)  # shape 10,15
    b2 = np.random.rand(10)  # shape  10

    for x in array_trainx:
        # x shape is 784,1
        z1 = np.dot(W1, (x)) + b1
        h1 = sigmoid(z1)
        z2 = np.dot(W2, h1) + b2
        h2 = softmax(z2)


if __name__ == '__main__':
    main()
