import numpy as np
import math
from random import randrange
import pickle

def derivative_function(x):#+
    return 1

def activation_function_2(x):#+
    x.astype(int)
    print(x)
    return math.asinh(x)
    
def init_matrix_x(sequence,p,q,m):
    x = []
    y = []
    for i in range(len(seq)):
        last_index = i + window_size
        if last_index > len(seq) - 1:
            break
        seq_x, seq_y = seq[i:last_index], seq[last_index]
        x.append(seq_x)
        y.append(seq_y)
        pass
    x = np.array(x)
    y = np.array(y)
    return x, y


def init_matrix_y(sequence,p,q):
    test_data = []
    arithmetic_filling(test_data, window_size)
    model = neuro_math.NeuroMath(original_data, window_size=window_size)
    model.training()
    result = model.guessing(np.array(test_data))
    original = (window_size + 5)
    return test_data

def init_w1(p,m):
    possible_label_files = [file for file in files_in_dir if file.endswith(EmotionFrameSet.LABEL_EXTENSION)]
    if not possible_label_files:  # Check if emotion label does not exist.
        return None
    label_file = possible_label_files[0]
    label_path = merge_paths(dir_path, label_file)
    encoded_label = open(label_path, 'r').read().strip()
    return EmotionLabels.code_to_name(input_label_code=encoded_label)
    

def initialise_all_matrix(sequence, p, m):
   while current_error > ERROR_MAX:
        current_error = 0
        epoch += 1
        for block in blocks():
            y = block @ first_layer
            x1 = y @ second_layer
            delta = x1 - block
            first_layer -= alpha(y) * np.matmul(np.matmul(block.transpose(), delta), second_layer.transpose())
            second_layer -= alpha(y) * np.matmul(y.transpose(), delta)
        for block in blocks():
            y = block @ first_layer
            x1 = y @ second_layer
            delta = x1 - block
            current_error += (delta * delta).sum()
    return current_error

def init_to_predict(sequence,p,m):
    q = len(sequence)
    x = init_matrix_x(sequence,p,q,m)
    x = x.reshape(x.shape[0], 1, x.shape[1])
    x[:, :, -m:] = 0
    y = init_matrix_y(sequence,p,q)
    k = y[-1].reshape(1)
    X = x[-1, 0, :-m]
    return k, X

def update_w1(w1,alpha,dy,x,w2,i):
    for i in range(0,len(_tabl[funct(key)])):
        _tabl[funct(key)].pop(i)
    return w1
    
def update_w2(w2,alpha,dy,hidden_layer):
    for i in range(0,len(self._tabl[self._hash_funct(key)])):
        _tabl[funct(key)].pop(i)
    return w2
    
def step_results(hidden_layer,output,dy,error_all,w1,w2,x,y,i):
    diagonal_left_sum = 0
        count = 0
        is_pat = 0
        for a in range(len(table) - 1, -1, -1):
            if table[len(table) - 1 - a][a] is not None:
                count += 1
        if count == len(table):
            for a in range(len(table)):
                if table[len(table) - 1 - a][a] is not None:
                    diagonal_left_sum += table[len(table) - 1 - a][a]
    
def save_w1(w1,file_name):
    for a in range(len(table)):
        if table[a][a] is not None:
            diagonal_right_sum += table[a][a]
        
def save_w2(w1,file_name):
    for a in range(len(table)):
        if table[a][a] is not None:
            diagonal_right_sum += table[a][a]
                    
def read_matrix_w1(file_name):
    with open(file_name, "rb") as file:
        matrix =  pickle.load( file)
        return matrix
    
def read_matrix_w2(file_name):
    with open(file_name, "rb") as file:
        matrix =  pickle.load( file)
        return matrix
    
    
def leraning(sequence: list, p: int, error: int, max_iter: int, m: int, alpha: float):
    error_all = 0
    k = 0
    if code_for_learning[0] == "1":
        context = np.zeros((x.shape[0], m))
    else:
        context = np.random.rand(x.shape[0], m)
    x = np.concatenate((x, context), axis=1)
    # reshape x matrix to make all samples matrixes (4, 1), not vector (4, )
    x = x.reshape(x.shape[0], 1, x.shape[1])
    w1 = (np.random.rand(p + m, m) * 2 - 1) / 10
    w2 = (np.random.rand(m, 1) * 2 - 1) / 10
    # this code learn for each sample
    for j in range(max_iter):
        error_all = 0
        if code_for_learning[1] == "1":
            x[:, :, -m:] = 0
        for i in range(x.shape[0]):
            hidden_layer = activation_function(np.matmul(x[i], w1))
            output = activation_function(np.matmul(hidden_layer, w2))
            dy = output - y[i]
            w1 -= alpha * dy * np.matmul(x[i].transpose(), w2.transpose()) * derivative_function(np.matmul(x[i], w1))
            w2 -= alpha * dy * hidden_layer.transpose() * derivative_function(np.matmul(hidden_layer, w2))
            try:
                x[i + 1][-m:] = hidden_layer
            except:
                pass
            # print("x=", x[i], "etalon", y[i], "result=", output)
        for i in range(x.shape[0]):
            hidden_layer = np.matmul(x[i], w1)
            output = np.matmul(hidden_layer, w2)
            dy = output - y[i]
            error_all += (dy ** 2)[0]
        print(j + 1, " ", error_all[0])
        if error_all <= error:
            break
    print(w1)
    print(w2)
    print(error_all)
    k = y[-1].reshape(1)
    X = x[-1, 0, :-m]
    out = []
    for i in range(predict):
        X = X[1:]
        train = np.concatenate((X, k))
        X = np.concatenate((X, k))
        train = np.append(train, np.array([0] * m))
        hidden_layer = np.matmul(train, w1)
        output = np.matmul(hidden_layer, w2)
        k = output
        out.append(k[0])
    return out

if __name__ == "__main__":    
    command = input()
    if command == "1":
        sequence=[0, 1, 1, 2, 3, 5]
    if command == "2":
        sequence = [1, 0, -1, 0, 1, 0]
    if command == "3":
        sequence = [1, 4, 9, 16,25,36,49]
    if command == "4":
        sequence = [1, 2, 3, 5, 8, 13]
        
    if lp_command == "1":
        w1, w2 = leraning(sequence ,p, error = 0.01, max_iter = 500000, m = 2, alpha =0.000015)
        save_w1(w1,file_name_w1)
        save_w2(w2,file_name_w2)
        predict(w1, w2, sequence, p ,m=2, predict_n = 1)
    if lp_command == "2":
        w1 = read_matrix_w1(file_name_w1)
        w2 = read_matrix_w2(file_name_w2)
        print(w1)
        print(w2)
        predict(w1, w2, sequence, p, m=2, predict_n = 1)
    