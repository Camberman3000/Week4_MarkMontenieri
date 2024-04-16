# Mark Montenieri - MS548
# Week 4 project - Estimated time to complete - 1 hours
# Actual time to complete - 1 hours

import numpy as np
import scipy
import torch


def task_numpy():
    # NumPy
    num_list = [2, 4, 6, 8, 10]
    array = np.array(num_list)
    print("Numpy array: ", array)
    menu_loop()


def task_scipy():
    # SciPy

    print("SciPy Version: " + scipy.__version__)
    menu_loop()


def task_pytorch():
    # PyTorch
    ndarray = np.array([5, 155, 289])
    t = torch.from_numpy(ndarray)
    print(t)
    menu_loop()


def menu_loop():  # Main menu
    try:
        print("\nWelcome to Mark's Python AI Project.")
        print("Menu")
        print("1. Numpy")
        print("2. TensorFlow")
        print("3. PyTorch")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")
        if choice.isdigit() and 1 <= int(choice) <= 4:
            choice = int(choice)
            if choice == 1:
                task_numpy()
            elif choice == 2:
                task_scipy()
            elif choice == 3:
                print("CHOICE ADD")
                task_pytorch()
            elif choice == 4:
                print("Exiting program")
                quit()
        else:
            print("Invalid choice. Please choose a number between 1 and 4")
            menu_loop()  # Reload Menu
    except TypeError:
        print("Try Exception")


if __name__ == "__main__":
    menu_loop()  # Load Menu
