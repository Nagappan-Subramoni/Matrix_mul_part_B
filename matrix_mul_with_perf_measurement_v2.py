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

# mpiexec -n 16 python matrix_mul_with_perf_measurement_v2.py 500

from mpi4py import MPI
import time
import random
import sys


def create_matrix(rows, cols):
    return [[random.randint(-100, 100) for _ in range(cols)] for _ in range(rows)]


def matrix_mul_serial(a, b):
    a_rows, a_cols = len(a), len(a[0])
    b_rows, b_cols = len(b), len(b[0])
    result = [[0 for _ in range(b_cols)] for _ in range(a_rows)]
    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(a_cols):
                result[i][j] += a[i][k] * b[k][j]
    return result


def compute_element(a, b, i, j):
    return sum(a[i][k] * b[k][j] for k in range(len(b)))


def matrix_mul_parallel(A, B):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Track time spent in communication and computation
    bcast_start = MPI.Wtime()
    if rank == 0:
        m, n = len(A), len(A[0])
        n2, p = len(B), len(B[0])
        assert n == n2, "Incompatible matrix sizes"
        tasks = [(i, j) for i in range(m) for j in range(p)]
    else:
        A = B = tasks = None

    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)
    tasks = comm.bcast(tasks, root=0)
    bcast_end = MPI.Wtime()
    bcast_time = bcast_end - bcast_start

    # Task distribution
    chunk_size = len(tasks) // size
    extra = len(tasks) % size
    start = rank * chunk_size + min(rank, extra)
    end = start + chunk_size + (1 if rank < extra else 0)
    local_tasks = tasks[start:end]

    # Computation
    comp_start = MPI.Wtime()
    local_results = []
    for i, j in local_tasks:
        val = compute_element(A, B, i, j)
        local_results.append((i, j, val))
    comp_end = MPI.Wtime()
    comp_time = comp_end - comp_start

    # Gather timing
    gather_start = MPI.Wtime()
    gathered = comm.gather(local_results, root=0)
    gather_end = MPI.Wtime()
    gather_time = gather_end - gather_start

    total_comm_time = bcast_time + gather_time

    print(f"[Rank {rank}] Broadcast time: {bcast_time:.6f}, Gather time: {gather_time:.6f}, Computation time: {comp_time:.6f}")

    if rank == 0:
        m, p = len(A), len(B[0])
        result = [[0] * p for _ in range(m)]
        for part in gathered:
            for i, j, val in part:
                result[i][j] = val
        return result
    return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Accept matrix size from command-line (default 500x500)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    # Only root creates matrices and measures serial performance
    if rank == 0:
        print(f"\nMatrix size: {N}x{N}")
        A = create_matrix(N, N)
        B = create_matrix(N, N)

        print("Measuring serial execution time...")
        start_serial = time.time()
        serial_result = matrix_mul_serial(A, B)
        end_serial = time.time()
        serial_time = end_serial - start_serial
        print(f"Serial execution time: {serial_time:.4f} seconds")

    else:
        A = None
        B = None

    # Start parallel execution
    if rank == 0:
        print("Measuring parallel execution time...")
    start_parallel = time.time()
    parallel_result = matrix_mul_parallel(A, B)
    end_parallel = time.time()

    if rank == 0:
        parallel_time = end_parallel - start_parallel
        print(f"Parallel execution time: {parallel_time:.4f} seconds")

        # Validate correctness
        is_correct = parallel_result == serial_result
        print(f"Results match: {is_correct}")

        # Speedup
        speedup = serial_time / parallel_time
        print(f"Speedup: {speedup:.2f}x with {comm.Get_size()} processes")
