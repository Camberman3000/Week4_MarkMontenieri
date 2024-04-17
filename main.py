# Mark Montenieri - MS548
# Week 4 project - Estimated time to complete - 1 hours
# Actual time to complete - 1 hours

import numpy as np
import scipy
import torch


def task_numpy():
    # NumPy
    num_list = [2, 4, 6, 8, 10]
    arr = np.array(num_list)
    print("Numpy array: ", arr)
    print("Number of dimensions: ", arr.ndim)
    print("Shape of the array: ", arr.shape)
    print("Size of the array: ", arr.size)
    print("Array element type: ", arr.dtype)
    menu_loop()


def task_scipy():
    # SciPy
    print("SciPy Version: " + scipy.__version__)
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    mean_val = np.mean(data)  # calc mean
    std_dev = np.std(data)  # calc std deviation
    print("The data mean:", mean_val)
    print("The data std deviation:", std_dev)
    menu_loop()


def task_pytorch():
    # PyTorch
    ndarray = np.array([5, 155, 289])
    t = torch.from_numpy(ndarray)
    print(t)

########### EXAMPLE FROM PYTORCH.ORG #############
    import math

    class LegendrePolynomial3(torch.autograd.Function):
        """
        We can implement our own custom autograd Functions by subclassing
        torch.autograd.Function and implementing the forward and backward passes
        which operate on Tensors.
        """

        @staticmethod
        def forward(ctx, input):
            """
            In the forward pass we receive a Tensor containing the input and return
            a Tensor containing the output. ctx is a context object that can be used
            to stash information for backward computation. You can cache arbitrary
            objects for use in the backward pass using the ctx.save_for_backward method.
            """
            ctx.save_for_backward(input)
            return 0.5 * (5 * input ** 3 - 3 * input)

        @staticmethod
        def backward(ctx, grad_output):
            """
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
            """
            input, = ctx.saved_tensors
            return grad_output * 1.5 * (5 * input ** 2 - 1)

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0")  # Uncomment this to run on GPU

    # Create Tensors to hold input and outputs.
    # By default, requires_grad=False, which indicates that we do not need to
    # compute gradients with respect to these Tensors during the backward pass.
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # Create random Tensors for weights. For this example, we need
    # 4 weights: y = a + b * P3(c + d * x), these weights need to be initialized
    # not too far from the correct result to ensure convergence.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
    b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
    c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
    d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 5e-6
    for t in range(2000):
        # To apply our Function, we use Function.apply method. We alias this as 'P3'.
        P3 = LegendrePolynomial3.apply

        # Forward pass: compute predicted y using operations; we compute
        # P3 using our custom autograd operation.
        y_pred = a + b * P3(c + d * x)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())

        # Use autograd to compute the backward pass.
        loss.backward()

        # Update weights using gradient descent
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            # Manually zero the gradients after updating weights
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

    print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')

    ########### END EXAMPLE FROM PYTORCH.ORG #############
    menu_loop()


def menu_loop():  # Main menu
    try:
        print("\nWelcome to Mark's Python AI Project.")
        print("Menu")
        print("1. Numpy")
        print("2. SciPy")
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
