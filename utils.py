import copy
import random

def period(a):
    if a % 2 == 0:
        return 0
    else:
        return 1

def period_sequence(n):
    return [period(i) for i in range(n)]

def degree_sequence(a, n):
    return [a ** i for i in range(n)]

def arithmetic_progression(a, n):
    return [a + i for i in range(n)]

def fib(n):
    a = 0
    b = 1
    result = []
    for __ in range(n):
        a, b = b, a + b
        result.append(a)
    return result

def activate(vector):

    # f(x)={0.01x, if x<0
    #       x, otherwise.}

    result_vector = []
    for i in range(len(vector)):
        if vector[i] >= 0:
            result_vector.append(vector[i])
        else:
            result_vector.append(0.01*vector[i])

    return result_vector

def function_derivative(vector):

    # f(x)={0.01x, if x<0
    #       x, otherwise.}
    result_vector = []
    for i in range(len(vector)):
        if vector[i] >= 0:
            result_vector.append(1)
        else:
            result_vector.append(0.01)

    return result_vector

def input_parametrs():
    p = int(input('Input p: '))
    L = int(input('Input L: '))

    alpha = float(input('Input alpha: '))
    e = float(input('Input square error: '))

    sequence = []

    menu = int(input('----------CHOOSE SEQUENCE----------\n\n1)Period\n2)Degree\n3)Arithmetic progression\n4)Fibonacci\n'))

    match(menu):
        case 1:
            a = int(input('Enter limit of sequence: '))
            sequence = period_sequence(a)
        case 2:
            a = int(input('Enter number: '))
            n = int(input('Enter limit of degree: '))
            sequence = degree_sequence(a, n)
        case 3:
            a = int(input('Enter number: '))
            n = int(input('Enter limit of progression: '))
            sequence = arithmetic_progression(a, n)
        case 4:
            n = int(input('Enter limit of sequence: '))
            sequence = fib(n)

    return p, L, alpha, e, sequence

def generate_matrix(sequence, p, L):
    return [[sequence[i+l] for i in range(p)] for l in range(L)]

def generate_weight_matrix(p, L):
    # матрица весов 1-го слоя
    matrix_1 = [[random.uniform(-1, 1) for j in range(L)] for i in range(p+L)]
    # матрица весов 2-го слоя
    matrix_2 = [[random.uniform(-1, 1) for j in range(1)] for i in range(L)]

    return matrix_1, matrix_2

def transpose(matrix):
    matrix = copy.deepcopy(matrix)
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]
    #транспонирование матрицы

def multiplication(a, b):
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    zip_b = zip(*b)
    zip_b = list(zip_b)  # массив столбцов второй матрицы

    return [[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b)) for col_b in zip_b] for row_a in a]

    # zip(row_a, col_b) - массив из пар элемента строки первой матрицы и элемента столбца второй матрицы
    # перемножаем эту пару, складываем

def m_sum(a, b):
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    # сумма матриц А и В

def m_diff(a, b):
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    #разность матриц А и В

def correct_weights(matrix_1, matrix_2,hidden, dX, alpha, input_matrix=[], Y=[], S=[], layer=1, vector=0):
    matrix_1 = copy.deepcopy(matrix_1)
    matrix_2 = copy.deepcopy(matrix_2)
    hidden = copy.deepcopy(hidden)
    dX = copy.deepcopy(dX)

    if layer == 1:
        save_matrix = multiplication(transpose([input_matrix]), transpose(matrix_2))
        save_matrix = multiplication(save_matrix, [function_derivative(S[0])])

        save_matrix = [[alpha * dX * save_matrix[i][j] for j in range(len(save_matrix[0]))] for i in
                       range(len(save_matrix))]

        return m_diff(matrix_1, save_matrix)

    else:
        save_matrix = multiplication(transpose(hidden), [function_derivative(Y[0])])
        save_matrix = [[alpha * dX * save_matrix[i][j] for j in range(len(save_matrix[0]))] for i in range(len(save_matrix))]

        return m_diff(matrix_2, save_matrix)

def step_training(i, input_matrix, context_matrix_elman, alpha, matrix_1, matrix_2):
    input = input_matrix[i] + context_matrix_elman

    S = multiplication([input], matrix_1)

    hidden = [activate(S[0])]
    Y = multiplication(hidden, matrix_2)

    output = [activate(Y[0])]

    dX = m_diff(output, [[input_matrix[i + 1][-1]]])[0][0]

    context_matrix_elman_new = hidden[0]

    save_matrix = matrix_2
    matrix_2 = correct_weights(matrix_1, matrix_2, hidden, dX, alpha, Y=Y, layer=2)
    matrix_1 = correct_weights(matrix_1, save_matrix, S, dX, alpha,  S=S, vector=i, input_matrix=input)

    return context_matrix_elman_new, context_matrix_elman, matrix_1, matrix_2, dX

def step_predict(i, input_matrix, context_matrix_elman, matrix_1, matrix_2):
    input = input_matrix[i] + context_matrix_elman

    S = multiplication([input], matrix_1)

    Y = multiplication(S, matrix_2)

    return Y[0][0]

def write_to_txt(matrix_1, matrix_2, context):
    # сохранение весов
    with open('matrix_1.txt', 'w') as f:
        for i in range(len(matrix_1)):
            for j in range(len(matrix_1[0])):
                f.write(f"{matrix_1[i][j]} ")
            f.write('\n')

    with open('matrix_2.txt', 'w') as f:
        for i in range(len(matrix_2)):
            for j in range(len(matrix_2[0])):
                f.write(f"{matrix_2[i][j]} ")
            f.write('\n')

    with open('context.txt', 'w') as f:
        for i in range(len(context)):
            for j in range(len(context[0])):
                f.write(f"{context[i][j]} ")
            f.write('\n')

