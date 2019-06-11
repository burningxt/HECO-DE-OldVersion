# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:56:03 2018

@author: burningxt
"""

import os
import numpy as np
import xlwt 
from xlwt import Workbook 

class PMODE_Results:
    cur_path = os.path.dirname(__file__)
    def getData_obj(self, benchID, D):
        os.listdir()
        new_path_1 = os.path.relpath('GECCO/Results/F{}_{}D_obj.txt'.format(benchID, D), self.cur_path)   
        Data_obj = []
        with open(new_path_1, 'r') as f:
            for line in f:
                Data_obj.append(float(line))
        return Data_obj
    
    def getData_vio(self, benchID, D):
        new_path_2 = os.path.relpath('GECCO/Results/F{}_{}D_vio.txt'.format(benchID, D), self.cur_path)   
        Data_vio = []
        with open(new_path_2, 'r') as f:
            for line in f:
                Data_vio.append(float(line))
        return Data_vio  
    
    def getData_c(self, benchID, D):
        new_path_3 = os.path.relpath('GECCO/Results/F{}_{}D_c.txt'.format(benchID, D), self.cur_path)   
        Data_c = []
        with open(new_path_3, 'r') as f:
            for line in f:
                Data_c.append(line)
        return Data_c 
    
class CMODE_Results:
    cur_path = os.path.dirname(__file__)
    def getData_obj(self, benchID, D):
        new_path_1 = os.path.relpath('GECCO/Results/F{}_{}D_obj.txt'.format(benchID, D), self.cur_path)   
        Data_obj = []
        with open(new_path_1, 'r') as f:
            for line in f:
                Data_obj.append(float(line))
        return Data_obj
    
    def getData_vio(self, benchID, D):
        new_path_2 = os.path.relpath('GECCO/Results/F{}_{}D_vio.txt'.format(benchID, D), self.cur_path)   
        Data_vio = []
        with open(new_path_2, 'r') as f:
            for line in f:
                Data_vio.append(float(line))
        return Data_vio     
    
    def getData_c(self, benchID, D):
        new_path_3 = os.path.relpath('GECCO/Results/F{}_{}D_c.txt'.format(benchID, D), self.cur_path)   
        Data_c = []
        with open(new_path_3, 'r') as f:
            for line in f:
                Data_c.append(line)
        return Data_c 

class PMODE_pfm:
    Dimensions = [10, 30, 50, 100]
    def __init__(self):
        PMODE_Data_obj = []
        PMODE_Data_vio = []        
        for D in self.Dimensions:
            temp = []
            for i in range(2):
                temp.append([])
            for benchID in range(1, 29):
                temp[0].append(PMODE_Results().getData_obj(benchID, D))
                temp[1].append(PMODE_Results().getData_vio(benchID, D))
            PMODE_Data_obj.append(temp[0])
            PMODE_Data_vio.append(temp[1])

        self.P_obj = PMODE_Data_obj
        self.P_vio = PMODE_Data_vio


              

class Compare_mean:
    Dimensions = [10, 30, 50, 100]
    def __init__(self):
        PMODE_Data_obj = []
        PMODE_Data_vio = []
        CMODE_Data_obj = []
        CMODE_Data_vio = []
        PMODE_Data_mean_obj = []
        PMODE_Data_mean_vio = []
        CMODE_Data_mean_obj = []
        CMODE_Data_mean_vio = []
        
        for D in self.Dimensions:
            temp = []
            for i in range(8):
                temp.append([])
            
            for benchID in range(1, 29):
                temp[0].append(sum(PMODE_Results().getData_obj(benchID, D)) / 25)
                temp[1].append(sum(CMODE_Results().getData_obj(benchID, D)) / 25)
                temp[2].append(sum(PMODE_Results().getData_vio(benchID, D)) / 25)
                temp[3].append(sum(CMODE_Results().getData_vio(benchID, D)) / 25)
                temp[4].append(PMODE_Results().getData_obj(benchID, D))
                temp[5].append(CMODE_Results().getData_obj(benchID, D))
                temp[6].append(PMODE_Results().getData_vio(benchID, D))
                temp[7].append(CMODE_Results().getData_vio(benchID, D))
            PMODE_Data_mean_obj.append(temp[0])
            CMODE_Data_mean_obj.append(temp[1])
            PMODE_Data_mean_vio.append(temp[2])
            CMODE_Data_mean_vio.append(temp[3])
            PMODE_Data_obj.append(temp[4])
            CMODE_Data_obj.append(temp[5])
            PMODE_Data_vio.append(temp[6])
            CMODE_Data_vio.append(temp[7])

        PMODE_fea = []
        for D in range(len(self.Dimensions)):
            temp = []
            for benchID in range(1, 29):
                fea_count_1 = 0
                for j in range(25):
                   if PMODE_Data_vio[D][benchID - 1][j] < 10.0**-100:
                       fea_count_1 += 1
                temp.append(fea_count_1 / 25) 
            PMODE_fea.append(temp)
        
        CMODE_fea = []

        for D in range(len(self.Dimensions)):
            temp = []
            for benchID in range(1, 29):
                fea_count_2 = 0
                for j in range(25):
                   if CMODE_Data_vio[D][benchID - 1][j] < 10.0**-100:
                       fea_count_2 += 1
                temp.append(fea_count_2 / 25) 
            CMODE_fea.append(temp)
#        print(PMODE_fea[0])
        self.P_obj = PMODE_Data_mean_obj
        self.P_vio = PMODE_Data_mean_vio
        self.P_fea = PMODE_fea
        self.C_obj = CMODE_Data_mean_obj
        self.C_vio = CMODE_Data_mean_vio 
        self.C_fea = CMODE_fea
#        for D in range(len(self.Dimensions)):
#                for benchID in range(1, 29):
#                    print(benchID, len(PMODE_Data_obj[D][benchID - 1]))
    def rankFea(self, benchID, D):
        if round(self.P_fea[D][benchID - 1], 8) > round(self.C_fea[D][benchID - 1], 8):
            jugg = 1
        elif round(self.P_fea[D][benchID - 1], 8) == round(self.C_fea[D][benchID - 1], 8):
            jugg = 0
        elif round(self.P_fea[D][benchID - 1], 8) < round(self.C_fea[D][benchID - 1], 8):
            jugg = -1
        return jugg

    def rankVio(self, jugg_fea, benchID, D):
        if jugg_fea == 0:
            if round(self.P_vio[D][benchID - 1], 8) < round(self.C_vio[D][benchID - 1], 8):
                jugg_vio = 1
            elif round(self.P_vio[D][benchID - 1], 8) == round(self.C_vio[D][benchID - 1], 8):
                jugg_vio = 0
            elif round(self.P_vio[D][benchID - 1], 8) > round(self.C_vio[D][benchID - 1], 8):
                jugg_vio= -1
        else:
            jugg_vio = None
        return jugg_vio
    
    def rankObj(self, jugg_vio, benchID, D):
        if jugg_vio == 0:
            if round(self.P_obj[D][benchID - 1], 8) < round(self.C_obj[D][benchID - 1], 8):
                jugg_obj = 1
            elif round(self.P_obj[D][benchID - 1], 8) == round(self.C_obj[D][benchID - 1], 8):
                jugg_obj = 0
            elif round(self.P_obj[D][benchID - 1], 8) > round(self.C_obj[D][benchID - 1], 8):
                jugg_obj= -1
        else:
            jugg_obj = None
        return jugg_obj

    def rankall(self):
        pos_fea = []
        equ_fea = []
        neg_fea = [] 
        pos_vio = []
        equ_vio = []
        neg_vio = [] 
        pos_obj = []
        equ_obj = []
        neg_obj = [] 
        pos_sum = []
        equ_sum = []
        neg_sum = [] 
        
        for D in self.Dimensions:           
            pos_fea.append([])
            equ_fea.append([])
            neg_fea.append([])
            pos_vio.append([])
            equ_vio.append([])
            neg_vio.append([])
            pos_obj.append([])
            equ_obj.append([])
            neg_obj.append([])
            pos_sum.append([])
            equ_sum.append([])
            neg_sum.append([])
        
        for D in range(len(self.Dimensions)):
            for benchID in range(1, 29):
                jugg_fea = self.rankFea(benchID, D)
                if jugg_fea == 1:
                    pos_fea[D].append(benchID)
#                    print('mean feasibility: {} {} +'.format(self.P_fea[D][benchID - 1], self.C_fea[D][benchID - 1]))
                elif jugg_fea == 0:
                    equ_fea[D].append(benchID)
#                    print('mean feasibility: {} {} ='.format(self.P_fea[D][benchID - 1], self.C_fea[D][benchID - 1]))
                elif jugg_fea == -1:
                    neg_fea[D].append(benchID)
#                    print('mean feasibility: {} {} -'.format(self.P_fea[D][benchID - 1], self.C_fea[D][benchID - 1]))
                    
                    
                jugg_vio = self.rankVio(jugg_fea, benchID, D)
                if jugg_vio == 1:
                    pos_vio[D].append(benchID)
#                    print('mean vio: {} {} +'.format(self.P_vio[D][benchID - 1], self.C_vio[D][benchID - 1]))
                elif jugg_vio == 0:
                    equ_vio[D].append(benchID)
#                    print('mean vio: {} {} ='.format(self.P_vio[D][benchID - 1], self.C_vio[D][benchID - 1]))
                elif jugg_vio == -1:
                    neg_vio[D].append(benchID)
#                    print('mean vio: {} {} -'.format(self.P_vio[D][benchID - 1], self.C_vio[D][benchID - 1]))
                    
                    
                jugg_obj = self.rankObj(jugg_vio, benchID, D)
                if jugg_obj == 1:
                    pos_obj[D].append(benchID)
#                    print('mean obj: {} {} +'.format(self.P_obj[D][benchID - 1], self.C_obj[D][benchID - 1]))
                elif jugg_obj == 0:
                    equ_obj[D].append(benchID)
#                    print('mean obj: {} {} ='.format(self.P_obj[D][benchID - 1], self.C_obj[D][benchID - 1]))
                elif jugg_obj == -1:
                    neg_obj[D].append(benchID)
#                    print('mean obj: {} {} -'.format(self.P_obj[D][benchID - 1], self.C_obj[D][benchID - 1]))
                
                pos_sum[D] = pos_fea[D] + pos_vio[D] + pos_obj[D] 
                equ_sum[D] = equ_obj[D]
                neg_sum[D] = neg_fea[D] + neg_vio[D] + neg_obj[D]
        return [pos_fea, equ_fea, neg_fea, pos_vio, equ_vio, neg_vio, pos_obj, equ_obj, neg_obj, pos_sum, equ_sum, neg_sum]

class Compare_median:
    Dimensions = [10, 30, 50, 100]
    def __init__(self):
        PMODE_Data_obj = []
        PMODE_Data_vio = []
        CMODE_Data_obj = []
        CMODE_Data_vio = []
        PMODE_Data_mean_obj = []
        PMODE_Data_mean_vio = []
        PMODE_Data_median_c = []
        CMODE_Data_mean_obj = []
        CMODE_Data_mean_vio = []
        CMODE_Data_median_c = []
        PMODE_Data_best_obj = []
        CMODE_Data_best_obj = []
        PMODE_Data_best_vio = []
        CMODE_Data_best_vio = []
        PMODE_Data_worst_obj = []
        CMODE_Data_worst_obj = []
        PMODE_Data_worst_vio = []
        CMODE_Data_worst_vio = []
#        for benchID in range(1, 29):
#            print(len(PMODE_Results().getData_vio(benchID, 10)))
        for D in self.Dimensions:
            temp = []
            for i in range(18):
                temp.append([])
                
            
            for benchID in range(1, 29):   
                ind_P = []
                ind_C = []
                for i in range(25):
#                    print(PMODE_Results().getData_c(benchID, D)[i])
                    ind_P.append([])
                    ind_P[i].append((PMODE_Results().getData_vio(benchID, D))[i])
                    ind_P[i].append((PMODE_Results().getData_obj(benchID, D))[i])
                    ind_P[i].append((PMODE_Results().getData_c(benchID, D))[i])
                    ind_C.append([])
                    ind_C[i].append((CMODE_Results().getData_vio(benchID, D))[i])
                    ind_C[i].append((CMODE_Results().getData_obj(benchID, D))[i])
                    ind_C[i].append((CMODE_Results().getData_c(benchID, D))[i])
                sorted_ind_P = sorted(ind_P, key=lambda x: (x[0],x[1]))
                sorted_ind_C = sorted(ind_C, key=lambda x: (x[0],x[1]))
#                print(sorted_ind_P)  
                
            
            
                temp[0].append(sorted_ind_P[12][1])
                temp[1].append(sorted_ind_C[12][1])
                temp[2].append(sorted_ind_P[12][0])
                temp[3].append(sorted_ind_C[12][0])
                temp[4].append(PMODE_Results().getData_obj(benchID, D))
                temp[5].append(CMODE_Results().getData_obj(benchID, D))
                temp[6].append(PMODE_Results().getData_vio(benchID, D))
                temp[7].append(CMODE_Results().getData_vio(benchID, D))
                temp[8].append(sorted_ind_P[0][1])
                temp[9].append(sorted_ind_C[0][1])
                temp[10].append(sorted_ind_P[0][0])
                temp[11].append(sorted_ind_C[0][0])
                temp[12].append(sorted_ind_P[24][1])
                temp[13].append(sorted_ind_C[24][1])
                temp[14].append(sorted_ind_P[24][0])
                temp[15].append(sorted_ind_C[24][0])
                temp[16].append(sorted_ind_P[12][2])
                temp[17].append(sorted_ind_C[12][2])
            PMODE_Data_mean_obj.append(temp[0])
            CMODE_Data_mean_obj.append(temp[1])
            PMODE_Data_mean_vio.append(temp[2])
            CMODE_Data_mean_vio.append(temp[3])
            PMODE_Data_obj.append(temp[4])
            CMODE_Data_obj.append(temp[5])
            PMODE_Data_vio.append(temp[6])
            CMODE_Data_vio.append(temp[7])
            PMODE_Data_best_obj.append(temp[8])
            CMODE_Data_best_obj.append(temp[9])
            PMODE_Data_best_vio.append(temp[10])
            CMODE_Data_best_vio.append(temp[11])
            PMODE_Data_worst_obj.append(temp[12])
            CMODE_Data_worst_obj.append(temp[13])
            PMODE_Data_worst_vio.append(temp[14])
            CMODE_Data_worst_vio.append(temp[15])
            PMODE_Data_median_c.append(temp[16])
            CMODE_Data_median_c.append(temp[17])

        PMODE_fea = []
        for D in range(len(self.Dimensions)):
            temp = []
            for benchID in range(1, 29):
                fea_count_1 = 0
                for j in range(25):
                   if PMODE_Data_vio[D][benchID - 1][j] < 10.0**-100:
                       fea_count_1 += 1
                temp.append(fea_count_1 / 25) 
            PMODE_fea.append(temp)
        
        CMODE_fea = []

        for D in range(len(self.Dimensions)):
            temp = []
            for benchID in range(1, 29):
                fea_count_2 = 0
                for j in range(25):
                   if CMODE_Data_vio[D][benchID - 1][j] < 10.0**-100:
                       fea_count_2 += 1
                temp.append(fea_count_2 / 25) 
            CMODE_fea.append(temp)
#        print(PMODE_Data_median_c)
        self.P_obj = PMODE_Data_mean_obj
        self.P_vio = PMODE_Data_mean_vio
        self.P_c = PMODE_Data_median_c
        self.C_c = CMODE_Data_median_c
        self.P_best_obj = PMODE_Data_best_obj
        self.P_best_vio = PMODE_Data_best_vio
        self.C_best_obj = CMODE_Data_best_obj
        self.C_best_vio = CMODE_Data_best_vio
        self.P_worst_obj = PMODE_Data_worst_obj
        self.P_worst_vio = PMODE_Data_worst_vio
        self.C_worst_obj = CMODE_Data_worst_obj
        self.C_worst_vio = CMODE_Data_worst_vio
        self.P_fea = PMODE_fea
        self.C_obj = CMODE_Data_mean_obj
        self.C_vio = CMODE_Data_mean_vio 
        self.C_fea = CMODE_fea
        
        
#        for D in range(len(self.Dimensions)):
#                for benchID in range(1, 29):
#                    print(benchID, len(PMODE_Data_obj[D][benchID - 1]))


    def rankVio(self, benchID, D):
        if round(self.P_vio[D][benchID - 1], 8) < round(self.C_vio[D][benchID - 1], 8):
            jugg_vio = 1
        elif round(self.P_vio[D][benchID - 1], 8) == round(self.C_vio[D][benchID - 1], 8):
            jugg_vio = 0
        elif round(self.P_vio[D][benchID - 1], 8) > round(self.C_vio[D][benchID - 1], 8):
            jugg_vio= -1
        
        return jugg_vio
    
    def rankObj(self, jugg_vio, benchID, D):
        if jugg_vio == 0:
            if round(self.P_obj[D][benchID - 1], 8) < round(self.C_obj[D][benchID - 1], 8):
                jugg_obj = 1
            elif round(self.P_obj[D][benchID - 1], 8) == round(self.C_obj[D][benchID - 1], 8):
                jugg_obj = 0
            elif round(self.P_obj[D][benchID - 1], 8) > round(self.C_obj[D][benchID - 1], 8):
                jugg_obj= -1
        else:
            jugg_obj = None
        return jugg_obj

    def rankall(self): 
        pos_vio = []
        equ_vio = []
        neg_vio = [] 
        pos_obj = []
        equ_obj = []
        neg_obj = [] 
        pos_sum = []
        equ_sum = []
        neg_sum = [] 
        
        for D in self.Dimensions:           
            pos_vio.append([])
            equ_vio.append([])
            neg_vio.append([])
            pos_obj.append([])
            equ_obj.append([])
            neg_obj.append([])
            pos_sum.append([])
            equ_sum.append([])
            neg_sum.append([])
        
        for D in range(len(self.Dimensions)):
            for benchID in range(1, 29):
                
                    
                jugg_vio = self.rankVio(benchID, D)
                if jugg_vio == 1:
                    pos_vio[D].append(benchID)
#                    print('mean vio: {} {} +'.format(self.P_vio[D][benchID - 1], self.C_vio[D][benchID - 1]))
                elif jugg_vio == 0:
                    equ_vio[D].append(benchID)
#                    print('mean vio: {} {} ='.format(self.P_vio[D][benchID - 1], self.C_vio[D][benchID - 1]))
                elif jugg_vio == -1:
                    neg_vio[D].append(benchID)
#                    print('mean vio: {} {} -'.format(self.P_vio[D][benchID - 1], self.C_vio[D][benchID - 1]))
                    
                    
                jugg_obj = self.rankObj(jugg_vio, benchID, D)
                if jugg_obj == 1:
                    pos_obj[D].append(benchID)
#                    print('mean obj: {} {} +'.format(self.P_obj[D][benchID - 1], self.C_obj[D][benchID - 1]))
                elif jugg_obj == 0:
                    equ_obj[D].append(benchID)
#                    print('mean obj: {} {} ='.format(self.P_obj[D][benchID - 1], self.C_obj[D][benchID - 1]))
                elif jugg_obj == -1:
                    neg_obj[D].append(benchID)
#                    print('mean obj: {} {} -'.format(self.P_obj[D][benchID - 1], self.C_obj[D][benchID - 1]))
                
                pos_sum[D] = pos_vio[D] + pos_obj[D] 
                equ_sum[D] = equ_obj[D]
                neg_sum[D] = neg_vio[D] + neg_obj[D]
        return [pos_vio, equ_vio, neg_vio, pos_obj, equ_obj, neg_obj, pos_sum, equ_sum, neg_sum]
  
    
if __name__ == '__main__':
    
    
    rank_results_mean = Compare_mean().rankall()
    pos_fea = rank_results_mean[0]
    equ_fea = rank_results_mean[1]
    neg_fea = rank_results_mean[2]
    pos_vio = rank_results_mean[3]
    equ_vio = rank_results_mean[4]
    neg_vio = rank_results_mean[5]
    pos_obj = rank_results_mean[6]
    equ_obj = rank_results_mean[7]
    neg_obj = rank_results_mean[8]
    pos_sum = rank_results_mean[9]
    equ_sum = rank_results_mean[10]
    neg_sum = rank_results_mean[11]
#    idx = 1
#    for D in [30]:
#        for benchID in pos_fea[idx]:
#            print('mean feasibility: f{}   {} {} +'.format(benchID, Compare_mean().P_fea[idx][benchID - 1], Compare_mean().C_fea[idx][benchID - 1]))
#        for benchID in neg_fea[idx]:
#            print('mean feasibility: f{}   {} {} -'.format(benchID, Compare_mean().P_fea[idx][benchID - 1], Compare_mean().C_fea[idx][benchID - 1]))
#        for benchID in equ_fea[idx]:
#            print('mean feasibility: f{}   {} {} ='.format(benchID, Compare_mean().P_fea[idx][benchID - 1], Compare_mean().C_fea[idx][benchID - 1]))
#        
#        for benchID in pos_vio[idx]:
#            print('mean vio: f{}   {} {} +'.format(benchID, Compare_mean().P_vio[idx][benchID - 1], Compare_mean().C_vio[idx][benchID - 1]))
#        for benchID in neg_vio[idx]:
#            print('mean vio: f{}   {} {} -'.format(benchID, Compare_mean().P_vio[idx][benchID - 1], Compare_mean().C_vio[idx][benchID - 1]))
#        for benchID in equ_vio[idx]:
#            print('mean vio: f{}   {} {} ='.format(benchID, Compare_mean().P_vio[idx][benchID - 1], Compare_mean().C_vio[idx][benchID - 1]))
#        
#        for benchID in pos_obj[idx]:
#            print('mean obj: f{}   {} {} +'.format(benchID, Compare_mean().P_obj[idx][benchID - 1], Compare_mean().C_obj[idx][benchID - 1]))
#        for benchID in neg_obj[idx]:
#            print('mean obj: f{}   {} {} -'.format(benchID, Compare_mean().P_obj[idx][benchID - 1], Compare_mean().C_obj[idx][benchID - 1]))
#        for benchID in equ_obj[idx]:
#            print('mean obj: f{}   {} {} ='.format(benchID, Compare_mean().P_obj[idx][benchID - 1], Compare_mean().C_obj[idx][benchID - 1]))
#        
#        print('in {}D      +: {},  =: {},  -:{}'.format(D, len(pos_sum[idx]), len(equ_sum[idx]), len(neg_sum[idx])))
#        idx += 1
 

    
    idx = 0
    for D in Compare_mean().Dimensions:
        file = open("mean_fea_{}D.txt".format(D), "w")
        for benchID in range(1, 29):
            file.write(str('%.8f'%round(100 * Compare_mean().C_fea[idx][benchID - 1], 8)) + '\n')
#            file.write(', ')
        file.close()
        file = open("mean_vio_{}D.txt".format(D), "w")
        for benchID in range(1, 29):
            file.write(str('%.8f'%round(Compare_mean().C_vio[idx][benchID - 1], 8)) + '\n')
#            file.write(', ')
        file.close()
        file = open("mean_obj_{}D.txt".format(D), "w")
        for benchID in range(1, 29):
            file.write(str('%.8f'%round(Compare_mean().C_obj[idx][benchID - 1], 8)) + '\n')
#            file.write(', ')
        file.close()
        idx += 1
    
    
    

      
#    rank_results_median = Compare_median().rankall()
#    pos_vio_median = rank_results_median[0]
#    equ_vio_median = rank_results_median[1]
#    neg_vio_median = rank_results_median[2]
#    pos_obj_median = rank_results_median[3]
#    equ_obj_median = rank_results_median[4]
#    neg_obj_median = rank_results_median[5]
#    pos_sum_median = rank_results_median[6]
#    equ_sum_median = rank_results_median[7]
#    neg_sum_median = rank_results_median[8]
#    idx = 0
#    for D in [10]:
#        for benchID in pos_vio_median[idx]:
#            print('median vio: f{}   {} {} +'.format(benchID, Compare_median().P_vio[idx][benchID - 1], Compare_median().C_vio[idx][benchID - 1]))
#        for benchID in neg_vio_median[idx]:
#            print('median vio: f{}   {} {} -'.format(benchID, Compare_median().P_vio[idx][benchID - 1], Compare_median().C_vio[idx][benchID - 1]))
#        for benchID in equ_vio_median[idx]:
#            print('median vio: f{}   {} {} ='.format(benchID, Compare_median().P_vio[idx][benchID - 1], Compare_median().C_vio[idx][benchID - 1]))
#        
#        for benchID in pos_obj_median[idx]:
#            print('median obj: f{}   {} {} +'.format(benchID, Compare_median().P_obj[idx][benchID - 1], Compare_median().C_obj[idx][benchID - 1]))
#        for benchID in neg_obj_median[idx]:
#            print('median obj: f{}   {} {} -'.format(benchID, Compare_median().P_obj[idx][benchID - 1], Compare_median().C_obj[idx][benchID - 1]))
#        for benchID in equ_obj_median[idx]:
#            print('median obj: f{}   {} {} ='.format(benchID, Compare_median().P_obj[idx][benchID - 1], Compare_median().C_obj[idx][benchID - 1]))
#        print('in {}D      +: {},  =: {},  -:{}'.format(D, len(pos_sum_median[idx]), len(equ_sum_median[idx]), len(neg_sum_median[idx])))
#        idx += 1

    
    
    idx = 0
    for D in Compare_median().Dimensions:
        file = open("median_vio_{}D.txt".format(D), "w")
        for benchID in range(1, 29):
            file.write(str('%.8f'%round(Compare_median().C_vio[idx][benchID - 1], 8)) + '\n')
#            file.write(', ')
        file.close()
        file = open("median_obj_{}D.txt".format(D), "w")
        for benchID in range(1, 29):
            file.write(str('%.8f'%round(Compare_median().C_obj[idx][benchID - 1], 8)) + '\n')
#            file.write(', ')
        file.close()
        idx += 1




   
    idx = 0
    for D in [10, 30, 50, 100]:
        wb = Workbook() 
        sheet1 = wb.add_sheet('Sheet 1') 
        sheet1.write(0, 0, 'problem') 
        sheet1.write(1, 0, 'Best') 
        sheet1.write(2, 0, 'Median')
        sheet1.write(3, 0, 'c')
        sheet1.write(4, 0, 'v')
        sheet1.write(5, 0, 'mean')
        sheet1.write(6, 0, 'Worst')
        sheet1.write(7, 0, 'std')
        sheet1.write(8, 0, 'SR')
        sheet1.write(9, 0, 'vio')
        
        for benchID in range(1, 29):
            sheet1.write(0, benchID, 'C0{}'.format(benchID))
            sheet1.write(1, benchID, "{0:.{1}e}".format(Compare_median().P_best_obj[idx][benchID - 1], 5))
            sheet1.write(2, benchID, "{0:.{1}e}".format(Compare_median().P_obj[idx][benchID - 1], 5))
            sheet1.write(3, benchID, "{}".format(Compare_median().P_c[idx][benchID - 1]))
            sheet1.write(4, benchID, "{0:.{1}e}".format(Compare_median().P_vio[idx][benchID - 1], 5))
            sheet1.write(5, benchID, "{0:.{1}e}".format(Compare_mean().P_obj[idx][benchID - 1], 5))
            sheet1.write(6, benchID, "{0:.{1}e}".format(Compare_median().P_worst_obj[idx][benchID - 1], 5))
            sheet1.write(7, benchID, "{0:.{1}e}".format(np.std(PMODE_pfm().P_obj[idx][benchID - 1]), 5))
            sheet1.write(8, benchID, "{}".format(int(100 * Compare_mean().P_fea[idx][benchID - 1])))
            sheet1.write(9, benchID, "{0:.{1}e}".format(Compare_mean().P_vio[idx][benchID - 1], 5))     
        wb.save('{}D.xls'.format(D)) 
        idx += 1





















