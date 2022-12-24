from utils import *

def train():
    p, L, alpha, e, sequence = input_parametrs()

    input_matrix = generate_matrix(sequence, p, L+1)
    matrix_1, matrix_2 = generate_weight_matrix(p, L)

    context_matrix_elman = [0 for _ in range(L)]
    iteration = 1

    while True:
        sum_error = 0

        for i in range(len(input_matrix)-1):
            context_matrix_elman, prev_elman, matrix_1, matrix_2,  dX = step_training(i, input_matrix, context_matrix_elman, alpha, matrix_1,
                                                                                                 matrix_2)
            sum_error += (dX ** 2)

        print(f'Error {iteration}: ', sum_error)
        iteration += 1

        if sum_error < e or iteration >=100000:
            write_to_txt(matrix_1, matrix_2, [prev_elman])
            break



