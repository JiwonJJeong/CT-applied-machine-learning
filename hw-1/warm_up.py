"""My code to do Part I. Warm Up"""
import numpy as np
import torch

# 1. Let a = np.array([1,2,3,4,5,6,7,8]). Reshape a into a 2 by 4 matrix.
def q1():
    a = np.array([1,2,3,4,5,6,7,8])
    a = np.array([a[0:4],a[4:8]])
    print(a.shape)

# 2. Let a be a pytorch tensor constructed with elements [1,3,5,6], and let b be a tensor constructed
# with elements [5, 6, 8, 9]. Write a sequence of lines, one each to perform the following operations
# on a and b. [Hint: use the function torch.tensor().]
# • Elementwise addition
# • Elementwise multiplication
# • Elementwise power (each element of a raised to the power given by corresponding element
# of b)
# • Dot product between a and b
# • Dot product between an elementwise exponentiation of a with base e and an elementwise
# natural logarithm of b
def q2():
    a = torch.tensor([1,3,5,6])
    b = torch.tensor([5,6,8,9])
    addition = a+b
    multiplication = a*b 
    power = torch.pow(a,b)
    dot1 = torch.dot(a,b)
    dot2 = torch.dot(torch.exp(a), torch.log(b))
    print(addition)
    print(multiplication)
    print(power)
    print(dot1)
    print(dot2)

# 3. Use tensor and autograd from the pytorch package to complete the following questions:
# (a) Calculate the gradient of
# g ¡x, y, z, k¢ = ex x2 + 3 e y y2 + 5ez z2 + 6ek k2
# evaluated at the point: ¡x = 5, y = 6, z = 8, k = 9¢.
# Hints: 1. you can rewrite the function g using the tensors [1, 3, 5, 6] and [5, 6, 8, 9] with
# correct tensor type; 2. set requires_grad to True for the correct tensor; 3. using the function
# ‘.backward()’; 4. obtain the gradient by calling ‘.grad’.

# g is same as input vector ([x, y, z, k] * [x, y, z, k] * [e^x, e^y, e^z, e^k]) • [1, 3, 5, 6]
                            # can rewrite x^2 operation as diagonal matrix
def q3_a():
    input = torch.tensor([5.,6.,8.,9.], requires_grad=True)
    g = torch.dot(torch.exp(input) * input * input, torch.tensor([1.,3.,5.,6.]))
    g.backward()
    gradient = input.grad
    print(gradient)

# (b) Let A be a matrix with values [[4, 3], [7, 9]] and B be a matrix with values [[3, 5], [1, 11]]. Calcu-
# late the gradient of the following function f(A) with respect to the entries of A evaluated at
# the point where A takes the above values.
# f (A) = log ∥AT AB T A AT AB ∥2
# In the above expression ∥ · ∥2 denotes the squared L2 norm, i.e., the sum of the squares of all
# entries of the matrix inside the norm expression.
# Hints: 1. to calculate matrix multiplication, you need to use the function torch.matmul; 2.
# to calculate L2 norm, you need to use the function torch.norm() and set p = 2.
def q3_b():
    a = torch.tensor([[4.,3.],[7.,9.]], requires_grad=True)
    b = torch.tensor([[3.,5.],[1.,11.]])
    # matrix multiplication ABC = (AB)C
    AB = torch.matmul(a,b)
    AAt = torch.matmul(a, torch.transpose(a,0,1))
    ABt = torch.matmul(a, torch.transpose(b,0,1))
    At = torch.transpose(a,0,1)
    inner_matrix_product = torch.matmul(At,torch.matmul(ABt, torch.matmul(AAt, AB)))
    norm = torch.norm((inner_matrix_product), p=2)
    f = torch.log(norm*norm)
    f.backward()
    print(a.grad)

# (c) Calculate the gradient of
# F ¡x, y¢ = tanh (x) + tanh ¡y¢
# at the point ¡x = 3, y = 7¢.

# tanhx = (expx - exp(-x))/(expx + exp(-x))
def q3_c():
    input = torch.tensor([3., 7.], requires_grad = True)
    f = torch.dot((torch.exp(input)-torch.exp(-1* input))/(torch.exp(input)+torch.exp(-1*input)), torch.tensor([1.,1.]))
    f.backward()
    gradient = input.grad
    print(gradient)

# 4. Let a be a torch integer tensor containing the values [1, 2, 3].
# 2
# CS5785 Fall 2025: Homework 1 Page 3
# • convert a to a numpy array and store it under a new variable b
# • convert a into a float tensor
def q4():
    a = torch.tensor([1,2,3])
    b = a.numpy()
    a = a.float()

# 5. Answer the following questions using the package Numpy:
# • What is the product of matrices of matrices [[1, 3, 5], [2, 1, 5]] and [[8, 4], [3, 6], [2, 7]]?
def q5_a():
    a = np.array([[1,3,5],[2,1,5]])
    b = np.array([[8,4],[3,6],[2,7]])
    print(np.matmul(a,b))

# • What is the Frobenius norm of the 1 x 3 matrix [100, 2, 1]?
def q5_b():
    a = np.array([100,2,1])
    print(np.linalg.norm(a))