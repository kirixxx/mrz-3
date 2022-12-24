from utils import *

def predict():
    # p, L, alpha, e, sequence = input_parametrs()



    # input_matrix = generate_matrix(sequence, p, L+1)

    with open('matrix_1.txt', 'r') as file:
        matrix_1 = file.readlines()
    matrix_1 = [[float(n) for n in x.split()] for x in matrix_1]

    with open('matrix_2.txt', 'r') as file:
        matrix_2 = file.readlines()
    matrix_2 = [[float(n) for n in x.split()] for x in matrix_2]

    with open('context.txt', 'r') as file:
        context_matrix_elman = file.readlines()
    context_matrix_elman = [[float(n) for n in x.split()] for x in context_matrix_elman][0]

    L = len(matrix_1[0])
    p = len(matrix_1)-L
    sequence = arithmetic_progression(2, 100)

    input_matrix = generate_matrix(sequence, p, L+1)
    print(input_matrix)

    for i in range(len(input_matrix)-1):
        output = step_predict(i, input_matrix, context_matrix_elman, matrix_1, matrix_2)
        print(output)
