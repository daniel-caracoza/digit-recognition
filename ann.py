"""
Daniel Caracoza
Machine Learning Fall 2019
Assignment 3 Artificial Neural Networks - Backpropagation Algorithm
"""
import pandas as pd
import numpy as np
from math import exp

EPOCH = 100
INSTANCES = 5000
TEST_INSTANCES = 9999
BIAS = 1
INPUT = 784
OUTPUT = 10
HIDDEN = 16
L_RATE = 0.04


def main():
    network = create_network()
    training_data = pd.read_csv("data\\training60000.csv")
    training_data_labels = pd.read_csv("data\\training60000_labels.csv").iloc[:, 0].tolist()
    output_weight_split = network[HIDDEN:len(network)]
    output_w_split_df = pd.DataFrame(output_weight_split)
    for j in range(EPOCH):
        for i in range(INSTANCES):
            x_i = np.asarray(training_data.iloc[i])
            x_i = np.insert(x_i, 0, BIAS)
            target_array = create_target_array(training_data_labels[i])
            # tuple of outputs (hidden, outputs)
            outputs = propagate_forward(x_i, network)
            deltas = propagate_backward(outputs, target_array, output_w_split_df)
            network = update_weights(x_i, deltas, outputs, network)
            network_error(target_array, outputs)

    test_accuracy(network)
# *
def sigmoid_func(net):
    return 1/(1 + exp(-net))

# *
def create_network():
    network = []
    for i in range(HIDDEN):
        # create weights for each hidden node
        network.append(np.random.uniform(-0.05, 0.05, INPUT + BIAS))
    # and each output node
    for j in range(OUTPUT):
        network.append(np.random.uniform(-0.05, 0.05, HIDDEN + BIAS))
    return network


# *
# need to keep track of h_outputs and o_outputs so maybe a separate list for both
def propagate_forward(input_vector, network):
    h_outputs = np.array([])
    o_outputs = np.array([])
    # list of hidden outputs- wi*xi -> sigmoid func -> h_outputs
    for i in range(HIDDEN):
        h_outputs = np.append(h_outputs, sigmoid_func(np.dot(input_vector, network[i])))
    # insert bias to the front of h_outputs
    h_outputs = np.insert(h_outputs, 0, 1)
    for j in range(OUTPUT):
        o_outputs = np.append(o_outputs, sigmoid_func(np.dot(h_outputs, network[HIDDEN + j])))
    return h_outputs, o_outputs


def propagate_backward(outputs, target_array, df):
    (h_outputs, o_outputs) = outputs
    delta_oks = delta_ok(o_outputs, target_array)  # *
    delta_hs = delta_h(h_outputs, delta_oks, df)
    return delta_hs, delta_oks


# *
# calculate the deltas for output layer
def delta_ok(o_outputs, target_array):
    delta_list = np.array([])
    for output, target_output in zip(o_outputs, target_array):
        delta_list = np.append(delta_list, output * (1-output) * (target_output - output))
    return delta_list


# *
# calculate deltas for hidden nodes by splitting network into columns of WeightsK_hi weights
# Turning the weight column into numpy array and dot product with delta_ok, this is done for every hidden_i
def delta_h(h_outputs, delta_oks, df):
    delta_hs = []
    # for every hidden node i, grab weight column Wk_hi and dot with delta_Ok
    # ex: delta_H1 = WO1_H1*delta_O1 + WO2_H1*delta_O2 ...
    for i in range(HIDDEN):
        # df.iloc[:,0] = weights for each output -> h0
        w_kh_i = df.iloc[:, i + 1]
        array_wk_hi = np.asarray(w_kh_i)
        # h_outputs[0] => h0 output which is => 1
        hk_error = h_outputs[i + 1] * (1 - h_outputs[i+1]) * np.dot(array_wk_hi, delta_oks)
        delta_hs.append(hk_error)
    return delta_hs


# wji = wji + (learning_rate)*delta_j*Xji
# ex: W_O1H0 = W_O1H0 + (learning_rate)*delta_O1*X_O1H0
def update_weights(input_vector, deltas,  outputs, network):
    (h_outputs, _) = outputs  # only need h_outputs
    (delta_hs, delta_oks) = deltas
    # traverse weights for hidden nodes
    for hidden_node in range(HIDDEN):
        for weight in range(INPUT + BIAS):
            network[hidden_node][weight] += L_RATE * delta_hs[hidden_node] * input_vector[weight]
    # traverse weights for outputs
    for output_node in range(OUTPUT):
        for weight in range(HIDDEN + BIAS):
            network[HIDDEN + output_node][weight] += L_RATE * delta_oks[output_node] * h_outputs[weight]
    return network

# *
def create_target_array(target_index):
    target_array = [0.01] * OUTPUT
    target_array[target_index] = 0.99
    return target_array

# *
def network_error(target_array, outputs):
    error = 0
    (_, o_outputs) = outputs  # only need o_outputs
    for i in range(len(target_array)):
        error += 0.5 * (target_array[i] - o_outputs[i]) ** 2
    print(error)

# *
def test_accuracy(network):
    test_data = pd.read_csv("data\\testing10000.csv")
    test_data_labels = pd.read_csv("data\\testing10000_labels.csv").iloc[:, 0].tolist()
    incorrect = 0
    correct = 0
    for i in range(TEST_INSTANCES):
        x_i = np.asarray(test_data.iloc[i])
        x_i = np.insert(x_i, 0, 1)  # Bias
        outputs = propagate_forward(x_i, network)
        target = test_data_labels[i]
        (_, o_outputs) = outputs  # only need o_outputs
        predicted = np.argmax(o_outputs)
        if predicted == target:
            correct += 1
        else:
            incorrect += 1

    print("Correct: ", correct)
    print("Incorrect: ", incorrect)
    print("Accuracy: ", float(correct / (correct + incorrect)))
    print("Epoch: ", EPOCH, ", HIDDEN: ", HIDDEN, ", Learning Rate: ", L_RATE, "Instances: ", INSTANCES)


main()

