# This code is written by Nagappan Subramoni nagappans@gmail.com
# 1) Mini-Project: MPI-Based Distributed Matrix Multiplication
# Project Title: Distributed Matrix Multiplication using MPI
# Objective: To implement and evaluate the performance of matrix multiplication across
# multiple nodes using MPI.
# Tasks:
# Environment Setup: Set up an MPI development environment.
# Matrix Multiplication: Implement a standard matrix multiplication algorithm.
# Distributed Implementation: Modify the algorithm for distributed computation using
# MPI, focusing on data partitioning and inter-process communication.
# Performance Metrics: Develop a system to measure execution time and scalability.
# Scalability Testing: Test the algorithm on different numbers of nodes/processes.
# Benchmarking: Benchmark against a serial implementation to evaluate performance
# gains.
# Deliverables: MPI-based distributed matrix multiplication code, performance metrics,
# benchmarking report, and detailed documentation.
#   python pymp_mat_mul.py <matrix size default 500> <Number of processors>
#  python pymp_mat_mul.py 400 16

import pymp
import numpy as np
import random
import time
import sys
import psutil

''' This method is used for computing one row and col value'''
def compute(a,b,row,col):
    return sum(a[row][k] * b[k][col] for k in range(len(b)))

''' This method is used for computing an array of  row and col value'''
def compute_chunk(a,b,rc):
    ret=[]
    for row,col in rc:
        ret.append((row, col, sum(a[row][k] * b[k][col] for k in range(len(b)))))
    return ret

''' Creating Empty matrix'''
def create_matrix(rows, cols):
    return [[random.randint(-100, 100) for _ in range(cols)] for _ in range(rows)]

''' Sequential Matrix MUl '''
def matrix_mul(a, b):
    a_rows, a_columns = len(a), len(a[0])
    b_rows, b_columns = len(b), len(b[0])
    result=[[0 for _ in range(a_columns)] for _ in range(a_rows)]
    for i in range(a_rows):
        for j in range(b_columns):
            for k in range(a_columns):
                result[i][j] += a[i][k] * b[k][j]
    return result

''' Matrix multiplication with one row and col at a time'''
def matrix_mul_par(a,b,proc=4):
    c_row = len(a)
    c_col = len(b[0])
    c = [[0 for _ in range(c_col)] for _ in range(c_row)]

    with pymp.Parallel(proc) as p:
        for row in range(c_row):
            for col in range(c_col):
                c[row][col] = compute(a, b, row, col)
    return c

''' Matrix multiplication with one batch of rows and cols '''
def matrix_mul_par2(a,b,proc=4):
    c_row = len(a)
    c_col = len(b[0])
    c = [[0 for _ in range(c_col)] for _ in range(c_row)]

    tasks=[]
    count=0
    tmp=[]
    for i in range(c_row):
        for j in range(c_col):
            tmp.append((i,j))
            count+=1
            if count % proc==0 :
                tasks.append(tmp)
                tmp=[]

    if(tmp):
        tasks.append(tmp)

    with pymp.Parallel(proc) as p:
        for task in tasks:
            result=compute_chunk(a,b,task)
            for row,col,val in result:
                c[row][col]=val

    return c

''' this function performs mat mul including performance metrics'''
def matrix_mul_par3(a, b, proc=4):
    c_row = len(a)
    c_col = len(b[0])
    c = [[0 for _ in range(c_col)] for _ in range(c_row)]

    # Task splitting
    tasks = []
    count = 0
    tmp = []
    for i in range(c_row):
        for j in range(c_col):
            tmp.append((i, j))
            count += 1
            if count % proc == 0:
                tasks.append(tmp)
                tmp = []

    if tmp:
        tasks.append(tmp)

    # Record CPU and memory info before computation
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().used

    # Start total timer
    start_time = time.perf_counter()

    ipc_times = []

    with pymp.Parallel(proc) as p:
        for task in tasks:
            # Simulate IPC delay timing (synchronization point between tasks)
            ipc_start = time.perf_counter()
            result = compute_chunk(a, b, task)
            ipc_end = time.perf_counter()
            ipc_times.append(ipc_end - ipc_start)

            for row, col, val in result:
                c[row][col] = val

    # End total timer
    end_time = time.perf_counter()

    # Record CPU and memory info after computation
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().used

    # Print metrics
    print("\n--- Performance Metrics ---")
    print(f"Total execution time      : {end_time - start_time:.6f} seconds")
    print(f"Average IPC (simulated)   : {sum(ipc_times) / len(ipc_times):.6f} seconds")
    print(f"Max IPC delay             : {max(ipc_times):.6f} seconds")
    print(f"Min IPC delay             : {min(ipc_times):.6f} seconds")
    print(f"CPU usage before          : {cpu_before}%")
    print(f"CPU usage after           : {cpu_after}%")
    print(f"Memory used               : {(mem_after - mem_before)/1e6:.2f} MB")

    return c

def canon1(A,B):
    r,c=A.shape
    q = r  # sqrt(number of processes)

    # Assume A_blocks and B_blocks are q x q numpy object arrays holding blocks
    A_blocks = np.copy(A)
    B_blocks = np.copy(B)


    # Initial alignment for A: left shift by row index
    for i in range(q):
        A_blocks[i] = np.roll(A_blocks[i], shift=-i, axis=0)  # left shift row i by i

    # Initial alignment for B: up shift by column index
    for j in range(q):
        B_blocks[:, j] = np.roll(B_blocks[:, j], shift=-j, axis=0)  # up shift col j by j

    return A_blocks, B_blocks

def canon2(A,B):

    A_blocks = np.copy(A)
    B_blocks = np.copy(B)
    A_blocks = np.roll(A_blocks, shift=-1, axis=1)  # Shift left (A)
    B_blocks = np.roll(B_blocks, shift=-1, axis=0)
    return A_blocks, B_blocks

def canon(A,B):
    r, c = A.shape
    C = np.zeros((r, c))
    A1, B1 = canon1(A, B)
    C += A1 * B1
    for i in range(r - 1):
        A1, B1 = canon2(A1, B1)
        C += A1 * B1
    return C


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    proc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    a=create_matrix(N,N)
    b=create_matrix(N,N)

    ax=np.array(a)
    bx=np.array(b)
    result=np.dot(ax,bx)

    start=time.time()
    seq=matrix_mul(a,b)
    end=time.time()
    print('Sequential time elapsed ',end-start)
    start=time.time()
    par=matrix_mul_par(a,b,proc)
    end=time.time()
    print('Parallel One Element at a time, time elapsed',end-start)

    print('Output of seq and parallel ',np.array_equal(result,par))

    start=time.time()
    par2=matrix_mul_par3(a,b,proc)
    end=time.time()
    print('Parallel one batch for each processor, time elapsed ',end-start)
    print('Output of seq and parallel ',np.array_equal(result,par2))

    start=time.time()
    par3=canon(ax,bx)
    end=time.time()
    print('Cannon time elapsed ',end-start)
    print('Output of seq and Cannon ',np.array_equal(result,par3))

