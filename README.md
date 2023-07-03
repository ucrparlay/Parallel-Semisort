# Parallel Algorithms for Semisort and Related Problems
This repository contains the code for our paper "High-Performance and Flexible Parallel Algorithms for Semisort and Related Problems". It includes the implementations for semisort, histogram, and collect-reduce. This repository is built on [ParlayLib](https://github.com/cmuparlay/parlaylib), we plan to integrate our code into the main branch of it. A full reference documentation of ParlayLib can be found [here](https://cmuparlay.github.io/parlaylib/).  

Prerequisite
--------
+ g++ or clang with C++17 features support (Tested with g++ 12.1.1 and clang 14.0.6) on Linux machines.

Getting Code
--------
The code can be downloaded using git:
```
git clone https://github.com/ucrparlay/Parallel-Semisort.git
```

Compilation
--------
Compilation can be done by using the Makefile in the ``include/parlay/`` directory. The ``make`` command compiles the ``semisort``, `histogram`, and ``collect_reduce``.
```
cd include/parlay/
make
```
It uses clang by default. To compile with g++, pass the flag ``GCC=1``.
```
make GCC=1
```

To compile our code to run in sequential, use: 
```
make SERIAL=1 
```

Running Code
--------
Simply run
```
./semisort [n]
```
where ``n`` is the input size. The data generator can be configured in the file ``semisort.cpp``.  

Contact
--------
If you have any questions, please submit an issue to this repository (recommended) or send an email to the author at xdong038@ucr.edu.  

Reference
--------
Xiaojun Dong, Yunshu Wu, Zhongqi Wang, Laxman Dhulipala, Yan Gu, Yihan Sun. [High-Performance and Flexible Parallel Algorithms for Semisort and Related Problems](https://dl.acm.org/doi/10.1145/3558481.3591071). In *ACM Symposium on Parallelism in Algorithms and Architectures (SPAA)*, pp. 341–353, 2023.  

Xiaojun Dong, Yunshu Wu, Zhongqi Wang, Laxman Dhulipala, Yan Gu, Yihan Sun. [High-Performance and Flexible Parallel Algorithms for Semisort and Related Problems](https://arxiv.org/abs/2304.10078). *arXiv preprint: 2304.10078*, 2023.  

If you use our code, please cite our paper:
```
@inproceedings{dong2023high,
  author    = {Dong, Xiaojun and Wu, Yunshu and Wang, Zhongqi and Dhulipala, Laxman and Gu, Yan and Sun, Yihan},
  title     = {High-Performance and Flexible Parallel Algorithms for Semisort and Related Problems},
  booktitle = {ACM Symposium on Parallelism in Algorithms and Architectures (SPAA)},
  year      = {2023},
}
```
