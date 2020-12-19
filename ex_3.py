import sys
import numpy as np
from scipy.special import softmax

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def main():
    learningrate = 0.15
    train_x = sys.argv[1]
    train_y = sys.argv[2]
    test_x = sys.argv[3]
    output = sys.argv[4]
    array_trainx = np.loadtxt(train_x)
    array_trainy = np.loadtxt(train_y)
    array_testx = np.loadtxt(test_x)
    array_testy = np.loadtxt(output)
    # normalize train_x
    colnumx = array_trainx.shape[1]
    rownumx = array_trainx.shape[0]

    # noramilizing
    for i in range(rownumx):
        for j in range(colnumx):
            array_trainx[i][j] = (array_trainx[i][j] / 255)
    rownumx = array_testx.shape[0]
    for i in range(rownumx):
        for j in range(colnumx):
            array_testx[i][j] = (array_testx[i][j] / 255)

    # Initialize random parameters and inputs
    temp = 15

    W1 = np.random.uniform(-1, 1, (temp, colnumx))  # shape 15,784
    b1 = np.random.uniform(-1, 1, (temp, 1))  # shape 15,1
    W2 = np.random.uniform(-1, 1, (10, temp))  # shape 10,15
    b2 = np.random.uniform(-1, 1, (10, 1))  # shape  10,1

    for x, y in zip(array_trainx, array_trainy):
        x = x.reshape(-1, 1)
        # x shape is 784,1
        z1 = np.dot(W1, x) + b1
        h1 = sigmoid(z1)
        z2 = np.dot(W2, h1) + b2
        h2 = softmax(z2)
        dz2 = h2
        dz2[int(y)] -= 1
        dW2 = np.dot(dz2, np.transpose(h1))  # dL/dz2 * dz2/dw2
        db2 = dz2  # dL/dz2 * dz2/db2
        dz1 = np.dot(np.transpose(W2),
                     (h2 - y)) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
        dW1 = np.dot(dz1, np.transpose(x))  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
        db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1

        #     # w1------4: h2-z2,z2-h1,h1-z1,z1-w1
        #     # h2-z2:
        #     h2[int(y)] = -1
        #     derivativeh2z2 = h2
        #     # z2=h1:
        #     derivativez2h1 = W2
        #     # h1-z1:
        #     derivativeh1z1 = sigmoid(z1) * (1 - sigmoid(z1))
        #     # z1-w1:
        #     derivativez1w1 = x
        #     derivativeh2w1 = np.dot(
        #         (np.dot(derivativeh2z2, derivativez2h1) * (derivativeh1z1.reshape(-1, 1)).transpose()).transpose(),
        #         derivativez1w1.reshape(-1, 1).transpose())
        #
        #     # w2
        #     derivativeh2w2 = np.dot(derivativeh2z2.reshape(-1, 1), h1.reshape(-1, 1).transpose())
        #     # b1
        #     derivativeh2b1 = (np.dot(derivativeh2z2, derivativez2h1) * (derivativeh1z1.reshape(-1, 1)).transpose())
        #     # b2
        #     derivativeh2b2 = derivativeh2z2

        W1 -= learningrate * dW1  # derivativeh2w1
        b2 -= learningrate * db2  # derivativeh2b2
        W2 -= learningrate * dW2  # derivativeh2w2
        b1 -= learningrate * db1  # ((derivativeh2b1))
    cntra = 0
    loss = 0
    for x, y in zip(array_testx, array_testy):
        x = x.reshape(-1, 1)
        z1 = np.dot(W1, (x)) + b1
        h1 = sigmoid(z1)
        z2 = np.dot(W2, h1) + b2
        h2 = softmax(z2)
        if y != np.argmax(h2):
            loss += 1
        cntra += 1
    print((loss / cntra) * 100)

    # for x in array_testx:
    #     x = x.reshape(-1, 1)
    #     z1 = np.dot(W1, (x)) + b1
    #     h1 = sigmoid(z1)
    #     z2 = np.dot(W2, h1) + b2
    #     h2 = softmax(z2)
    #     output = np.argmax(h2)
    #     print(str(output))


if __name__ == '__main__':
    main()
