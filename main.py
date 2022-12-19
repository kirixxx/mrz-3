import numpy as np
import math
from random import randrange
import pickle

def function_1(x):#+
    return 1

def function_2(x):#+
    x.astype(int)
    print(x)
    return math.asinh(x)
    
def matrix_init_1(matrix):#+
    serial = 0
    matrix = os.path.isfile("text2.txt")
    if matrix:
        with open("text1.txt", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    return matrix


def matrix_init_2(sequence,p,q):
    serial = 0
    matrix = os.path.isfile("text2.txt")
    if matrix:
        with open("text1.txt", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    return matrix

def w1(matrix,sq):
    for matrix, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': matrix,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        return w1
    

def init_matrixs(matrix, error):
   for current_error in SIGNATURE_CLASSES:
        fish_files = get_images(current_error)
        files.extend(fish_files)
    
    y_fish = np.tile(current_error, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), current_error))
    
    y_all = np.array(y_all)
    return current_error

def predict_init(matrix, s, error):
    predict = [im for im in os.listdir(TEST_DIR)]
    test = np.ndarray((len(predict), ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, im in enumerate(predict): 
    test[i] = read_image(TEST_DIR+im)
    
    test_preds = model.predict(test, verbose=1)
    submission = pd.DataFrame(test_preds, columns=SIGNATURE_CLASSES)
    submission.insert(0, 'image', predict)
    submission.head()
    return predict

def update_weight_1(w1, error, sq, alpha):
   model.fit(X_train, y_train, batch_size=64, nb_epoch=3,
        validation_split=0.1, verbose=1, shuffle=True, callbacks=[early_stopping])
    preds = model.predict(X_valid, verbose=1)
    return w1
    
def update_w2(w2,alpha,dy,hidden_layer):
    model.fit(X_train, y_train, batch_size=64, nb_epoch=3,
        validation_split=0.1, verbose=1, shuffle=True, callbacks=[early_stopping])
    preds = model.predict(X_valid, verbose=1)
    return w2
    
def steps(w1, w2, error, sq, matrix):
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
    
def save_weight_1(w1,file, matrix):
    for i in range(len(array)):
        for j in range(len(array)-1-i):
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
            update_display(array[j], array[j+1])

def save_weight_2(w2,file, matrix):
    for i in range(len(array)):
        min_idx = i
    for j in range(i+1, len(array)):
        if array[j] < array[min_idx]:
            min_idx = j
    array[i], array[min_idx] = array[min_idx], array[i]
    update_display(array[i], array[min_idx])
                    
def read_file(file_name):
    for i in range(len(array)):
        cursor = array[i]
        idx = i
        while idx > 0 and array[idx-1] > cursor:
            array[idx] = array[idx-1]
            idx -= 1
        array[idx] = cursor
        update_display(array[idx], array[i])
    
    
def start(sequence: list, p: int, error: int, max_iter: int, m: int, alpha: float):
    proc = preprocessing.LabelEncoder()
    sepal_length = proc.fit_transform(list(data["sepal_length"]))
    sepal_width = proc.fit_transform(list(data["sepal_width"]))
    petal_length = proc.fit_transform(list(data["petal_length"]))
    petal_width = proc.fit_transform(list(data["petal_width"]))
    variety = proc.fit_transform(list(data["species"]))

    # Prediction
    predict = "species"
    x = list(zip(sepal_length, sepal_width, petal_length, petal_width))
    y = list(variety)

    var = ['Setosa', 'Virginica', 'Versicolor']
    best = 0
    worst = 100

    for i in range(100):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, test_size=0.9)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        if accuracy > best:
            best = accuracy
        elif accuracy < worst:
            worst = accuracy
        prediction = model.predict(x_test)
        return out

if __name__ == "__main__":    
     if array == []:
        array = self.array
        end = len(array) - 1
    if start < end:
        pivot = self.partition(array,start,end)
        self.algorithm(array,start,pivot-1)
        self.algorithm(array,pivot+1,end)
    start(sequence ,p, error = 0.01, max_iter = 500000, m = 2, alpha =0.000015)