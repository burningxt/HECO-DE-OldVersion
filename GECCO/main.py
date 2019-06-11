# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:41:58 2018

@author: burningxt
"""

#from HEOEA_cy import HEOEA
#import GECCO_cy
from GECCO_cy import HEOEA
import timeit
import multiprocessing as mp
#
if __name__ == '__main__':
   start = timeit.default_timer()
   results = HEOEA(27, 10).optimize(27, 10)
   stop = timeit.default_timer()
   print('Time: ', stop - start)
        
#def run(runs):
#    for D in [100]:
#        for benchID in [6]:
#            print("testing benchmark function {} in {}D in {} run".format(benchID, D, runs + 1))
#            results = HEOEA(benchID, D).optimize(benchID, D)
#            print(results[0], file = open("Results/F{}_{}D_obj.txt".format(benchID, D), "a"))
#            print(results[1], file = open("Results/F{}_{}D_vio.txt".format(benchID, D), "a"))
#            print(results[2], results[3], results[4], file = open("Results/F{}_{}D_c.txt".format(benchID, D), "a")) 
#            
# 
#if __name__ == '__main__':
#    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
#    start = timeit.default_timer()
#    pool = mp.Pool(processes = 5)
#    res = pool.map(run, range(25))
#    stop = timeit.default_timer()
#    print('Time: ', stop - start) 
            
# def run(runs):
#     for D in [10, 30, 50, 100]:
#         for benchID in range(1, 29):
# #        for benchID in [25, 24, 12, 27, 4]:
#             results = HEOEA(benchID, D).optimize(benchID, D)
#             print(results[0], file = open("Results/F{}_{}D_obj.txt".format(benchID, D), "a"))
#             print(results[1], file = open("Results/F{}_{}D_vio.txt".format(benchID, D), "a"))
#             print(results[2], results[3], results[4], file = open("Results/F{}_{}D_c.txt".format(benchID, D), "a"))
#
#
# if __name__ == '__main__':
#     __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
#     start = timeit.default_timer()
#     pool = mp.Pool(processes = 25)
#     res = pool.map(run, range(25))
#     stop = timeit.default_timer()
#     print('Time: ', stop - start)