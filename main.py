from train import train
from predict import predict

while True:
    print("----------MENU----------")
    print("1) Training of neural network\n2) Predict\n3) Exit\n")
    menu = input("Enter number: ")

    if menu == '1':
        train()

    elif menu == '2':
        predict()

    elif menu == '3':
        exit()
