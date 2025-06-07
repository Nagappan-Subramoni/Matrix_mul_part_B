Short Description: The code is written with 4 steps with 2 solutions one using pymp and second using Open MP .
1. Sequential computation
2. Parallel execution , parallize each element in parallel
3. Parallel execution, batch of elements based on the number of processors, execution of all these chunks in parallel
4. For scalable purpose, implemented Cannon algorithm for parallel execution.

All the outputs are validated against standard np.dot numpy library. CPU statstics is measured only for Batch chunk processing.
Cannon algorithm implementation doesn't use parallel, when implemented with parallel execution got error in data due to sync.

Output:

(lvenv) nagappans@nagappan:/mnt/c/Users/nagappans/PycharmProjects/PythonProject$ python pymp_mat_mul.py 400
Sequential time elapsed  4.472261428833008
Parallel One Element at a time, time elapsed 4.588259935379028
Output of seq and parallel  True

--- Performance Metrics ---
Total execution time      : 4.340691 seconds
Average IPC (simulated)   : 0.000103 seconds
Max IPC delay             : 0.001948 seconds
Min IPC delay             : 0.000082 seconds
CPU usage before          : 15.1%
CPU usage after           : 24.5%
Memory used               : 9.42 MB
Parallel one batch for each processor, time elapsed  4.424896717071533
Output of seq and parallel  True
Cannon time elapsed  0.0968937873840332
Output of seq and Cannon  True
(lvenv) nagappans@nagappan:/mnt/c/Users/nagappans/PycharmProjects/PythonProject$ python pymp_mat_mul.py 400 8
Sequential time elapsed  5.185550689697266
Parallel One Element at a time, time elapsed 5.515494346618652
Output of seq and parallel  True

--- Performance Metrics ---
Total execution time      : 5.577070 seconds
Average IPC (simulated)   : 0.000269 seconds
Max IPC delay             : 0.001661 seconds
Min IPC delay             : 0.000176 seconds
CPU usage before          : 29.4%
CPU usage after           : 48.4%
Memory used               : 17.35 MB
Parallel one batch for each processor, time elapsed  5.64848256111145
Output of seq and parallel  True
Cannon time elapsed  0.10057997703552246
Output of seq and Cannon  True
(lvenv) nagappans@nagappan:/mnt/c/Users/nagappans/PycharmProjects/PythonProject$ python pymp_mat_mul.py 400 16
Sequential time elapsed  4.572847604751587
Parallel One Element at a time, time elapsed 7.948811054229736
Output of seq and parallel  True

--- Performance Metrics ---
Total execution time      : 8.217730 seconds
Average IPC (simulated)   : 0.000784 seconds
Max IPC delay             : 0.007505 seconds
Min IPC delay             : 0.000368 seconds
CPU usage before          : 61.9%
CPU usage after           : 95.4%
Memory used               : -0.07 MB
Parallel one batch for each processor, time elapsed  8.279032707214355
Output of seq and parallel  True
Cannon time elapsed  0.10022425651550293
Output of seq and Cannon  True
(lvenv) nagappans@nagappan:/mnt/c/Users/nagappans/PycharmProjects/PythonProject$ python pymp_mat_mul.py 500
Sequential time elapsed  9.446469783782959
Parallel One Element at a time, time elapsed 10.171149730682373
Output of seq and parallel  True

--- Performance Metrics ---
Total execution time      : 9.238656 seconds
Average IPC (simulated)   : 0.000147 seconds
Max IPC delay             : 0.001693 seconds
Min IPC delay             : 0.000104 seconds
CPU usage before          : 15.2%
CPU usage after           : 24.9%
Memory used               : 30.49 MB
Parallel one batch for each processor, time elapsed  9.380584239959717
Output of seq and parallel  True
Cannon time elapsed  0.24668169021606445
Output of seq and Cannon  True
(lvenv) nagappans@nagappan:/mnt/c/Users/nagappans/PycharmProjects/PythonProject$

Conclusion: Looks like the system is best operated when the number of processors are 4 and its efficacy goes down as we increase number of processors
The final Cannon implementation gives huge performance input and is good for scalable purpose.


Second Solution is using openmp solution by using windows


(.venv) C:\Users\nagappans\PycharmProjects\PythonProject>mpiexec -n 16 python matrix_mul_with_perf_measurement_v2.py 500
[Rank 4] Broadcast time: 12.055374, Gather time: 0.156151, Computation time: 1.293827
[Rank 5] Broadcast time: 12.068368, Gather time: 0.128198, Computation time: 1.308682
[Rank 6] Broadcast time: 12.080957, Gather time: 0.089798, Computation time: 1.335653
[Rank 7] Broadcast time: 12.059303, Gather time: 0.151906, Computation time: 1.295196
[Rank 9] Broadcast time: 12.072190, Gather time: 0.083614, Computation time: 1.350394
[Rank 8] Broadcast time: 12.071160, Gather time: 0.132136, Computation time: 1.303438
[Rank 2] Broadcast time: 12.060536, Gather time: 0.142586, Computation time: 1.301653
[Rank 12] Broadcast time: 12.062752, Gather time: 0.091625, Computation time: 1.353058
[Rank 11] Broadcast time: 12.055194, Gather time: 0.016856, Computation time: 1.434192
[Rank 10] Broadcast time: 12.050913, Gather time: 0.125543, Computation time: 1.330100
[Rank 13] Broadcast time: 12.077046, Gather time: 0.111995, Computation time: 1.319732
[Rank 1] Broadcast time: 12.054256, Gather time: 0.053433, Computation time: 1.396885
[Rank 14] Broadcast time: 12.058634, Gather time: 0.148720, Computation time: 1.300844
[Rank 3] Broadcast time: 12.058866, Gather time: 0.027857, Computation time: 1.419424
[Rank 15] Broadcast time: 12.054488, Gather time: 0.024733, Computation time: 1.448183

Matrix size: 500x500
Measuring serial execution time...
Serial execution time: 11.7199 seconds
Measuring parallel execution time...
[Rank 0] Broadcast time: 0.168636, Gather time: 0.072141, Computation time: 1.424129
Parallel execution time: 1.6822 seconds
Results match: True
Speedup: 6.97x with 16 processes

Conclusion: Using MPI the performance improvement is 6.96 times original time.
