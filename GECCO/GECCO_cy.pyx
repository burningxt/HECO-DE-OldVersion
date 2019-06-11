# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:02:35 2018

@author: burningxt
"""

from cec2017_cy import CFunction, Individual
#from cec2017 import CFunction, Individual
from sklearn.decomposition import PCA
from scipy.stats import truncnorm
from scipy.stats import cauchy
import numpy as np
import random
import operator
import timeit
from libc.math cimport sin, tan, pi, e, exp, sqrt, log
import multiprocessing as mp

cdef double rand_normal(double mu, double sigma):
    cdef:
        double z, uniform
    uniform = random.random()
    z = sqrt(- 2.0 * log(uniform)) * sin(2.0 * pi * uniform) 
    z = mu + sigma * z
    return z

cdef double rand_cauchy(double mu, double gamma):
    cdef:
        double z, uniform
    uniform = random.random()
    z = mu + gamma * tan(pi * (uniform - 0.5))
    return z

cdef class HEOEA:
    params = {'NP' : 12, 
              'H': 5,
              'num_stg': 4,
              'p_rate' : 0.11,
              'FES_MAX' : 20000
            }
    cdef public list lb
    cdef public list ub
    cdef public double[:, :] o 
    cdef public double[:, :] M 
    cdef public double[:, :] M1
    cdef public double[:, :] M2
    def __init__(self, benchID, D):
        self.lb = CFunction().getLowBound(benchID, D)
        self.ub = CFunction().getUpBound(benchID, D)
        mat = CFunction().loadmat(benchID, D)
        self.o = mat[0]
        self.M = mat[1]
        self.M1 = mat[2]
        self.M2 = mat[3]
    cdef list initPop(self, int NP, int benchID, int D):
        cdef:
            list P, X
        P = []
        for i in range(NP):
            X = []
            for j in range(D):
                X.append(self.lb[j] + (self.ub[j] - self.lb[j]) * random.random())
            P.append(Individual(benchID, D, X, self.o, self.M, self.M1, self.M2))
        return P
    
    cdef list initQ(self, list P, int NP, int Lambda):
        cdef:
            list Q, chosen_idx
            int inds
        Q = []
        _P = P[:]
        chosen_idx = random.sample(range(NP), Lambda)
        [Q.append(P[inds]) for inds in chosen_idx]
        [P.remove(_P[inds]) for inds in chosen_idx]
        return Q
        
    cdef list initMemory(self):
        cdef:
            int H, n, j
            list M_CR, M_F, Temp_1, Temp_2
        H = self.params['H']
        num_stg = self.params['num_stg']
        M_CR = []
        M_F = []
        for i in range(num_stg):
            
            Temp_1, Temp_2 = [], []
            for j in range(H - 1):
                Temp_1.append(0.5)
            Temp_1.append(0.5)
                
            M_CR.append(Temp_1)

            for j in range(H - 1):
                Temp_2.append(0.5)
            Temp_2.append(0.5)
                
            M_F.append(Temp_2)
        return [M_CR, M_F]
     
#    cdef int chooseStrategy(self, int fea_jugg, list ql, list Num_Success_n):
#        cdef:
#            int n_sum, k, l, Strategy, num_stg
#            double wheel
#        num_stg = self.params['num_stg']   
#        n_sum = 0
#        for k in range(num_stg):
#            n_sum += Num_Success_n[k] + 2
#        if n_sum != 0:
#            for k in range(num_stg):
#                ql[k] = <double>(Num_Success_n[k] + 2) / n_sum
#        for k in range(num_stg):
#            for l in range(num_stg):
#                if Num_Success_n[num_stg] > 100:
#                    Num_Success_n[num_stg] = 0
#            if ql[k] < 0.05:
#                for l in range(num_stg):
#                    ql[l] = 0.25
#                    Num_Success_n[l] = 0
#                break
#        wheel = random.random()
#        if wheel <= ql[0]:
#            Strategy = 0
#        elif wheel <= ql[0] + ql[1] and wheel > ql[0]:
#            Strategy = 1
#        elif wheel <= ql[0] + ql[1] + ql[2] and wheel > ql[0] + ql[1]:
#            Strategy = 2
#        elif wheel <= ql[0] + ql[1] + ql[2] + ql[3] and wheel > ql[0] + ql[1] + ql[2]:
#            Strategy = 3
#        return Strategy        
    cdef int chooseStrategy(self, list ql, list Num_Success_n):
        cdef:
            int n_sum, k, l, Strategy
            double wheel
            
        n_sum = 0
        Strategy = 0
        for k in range(4):
            n_sum += Num_Success_n[k] + 2
        if n_sum != 0:
            for k in range(4):
                ql[k] = <double>(Num_Success_n[k] + 2) / n_sum
        for k in range(4):
#            for l in range(4):
#                if Num_Success_n[l] > 100:
#                    Num_Success_n[l] = 0
            if ql[k] < 0.05:
                for l in range(4):
                    ql[l] = 0.25
                    Num_Success_n[l] = 0
                break
        wheel = random.random()
        if wheel <= ql[0]:
            Strategy = 0
        elif wheel <= ql[0] + ql[1] and wheel > ql[0]:
            Strategy = 1
        elif wheel <= ql[0] + ql[1] + ql[2] and wheel > ql[0] + ql[1]:
            Strategy = 2
        elif wheel <= ql[0] + ql[1] + ql[2] + ql[3] and wheel > ql[0] + ql[1] + ql[2]:
            Strategy = 3
        return Strategy    
#    cdef int chooseStrategy(self, list ql):
#        cdef:
#            int Strategy
#            double wheel
#        Strategy = 0
#        wheel = random.random()
#        if wheel <= ql[0]:
#            Strategy = 0
#        elif wheel <= ql[0] + ql[1] and wheel > ql[0]:
#            Strategy = 1
#        elif wheel <= ql[0] + ql[1] + ql[2] and wheel > ql[0] + ql[1]:
#            Strategy = 2
#        elif wheel <= ql[0] + ql[1] + ql[2] + ql[3] and wheel > ql[0] + ql[1] + ql[2]:
#            Strategy = 3
#        return Strategy

    cdef list generate_F_CR(self, int Stg, list Memory_CR, list Memory_F, list Success_CR, list CR, list F, double grate):
        cdef:
            int H, i, j, ri, num_stg
            double cr, f
            list muCR, muF
        H = self.params['H']
        num_stg = self.params['num_stg']
        ri = random.randint(0, H - 1)
        muCR = []
        for i in range(num_stg):
            muCR.append(0.5)
        muF = []
        for i in range(num_stg):
            muF.append(0.5)
        if Success_CR[Stg] != []:
            if muCR[Stg] == -1 or max(Success_CR[Stg]) == 0:
                muCR[Stg] == 0.0
            else:
                muCR[Stg] = Memory_CR[Stg][ri]
        else:
            muCR[Stg] = Memory_CR[Stg][ri]
        muF[Stg] = Memory_F[Stg][ri]
        cr = rand_normal(muCR[Stg], 0.1)
        f = rand_cauchy(muF[Stg], 0.1)
        
        if cr < 0:
            cr = 0
        elif cr > 1:
            cr = 1
        while f <= 0:
            f = rand_cauchy(muF[Stg], 0.1)
        if f > 1:
            f = 1
            
#        if grate < 0.25:
#            cr = max(cr, 0.5)
#        elif grate < 0.5:
#            cr = max(cr, 0.25)
#        if grate < 0.25:
#            f = min(f, 0.6)
#        elif grate < 0.5:
#            f = min(f, 0.7)
#        elif grate < 0.75:
#            f = min(f, 0.9)
        CR.append(cr)
        F.append(f) 
        return [cr, f]
    
    

    cdef list mutation_1(self, list Q, list A, int Lambda, int D, double f, int index, double grate, int jugg):
        cdef:
            double p_rate
            int x_r1, x_r2, i
            list PA, X, _Q
        p_rate = 0.11
        _Q = Q[:]
#        if jugg == -1 :
#            _Q = sorted(_Q, key=operator.attrgetter('obj'))
#            _Q = sorted(_Q, key=operator.attrgetter('vio'))
#        else:
#            _Q = sorted(_Q, key=operator.attrgetter('obj'))
        
        if jugg != -1 or grate < 0.4:
            _Q = sorted(_Q, key=operator.attrgetter('obj'))
        else:
            _Q = sorted(_Q, key=operator.attrgetter('obj'))
            _Q = sorted(_Q, key=operator.attrgetter('vio'))
#        _Q = sorted(_Q, key=operator.attrgetter('obj'))
        QA = []
        QA = Q[:] + A[:]
        x_r1 = random.randint(0, Lambda - 1)
        x_r2 = random.randint(0, Lambda + len(A) - 1)
        while x_r1 == index:
            x_r1 = random.randint(0, Lambda - 1)
        while x_r2 ==x_r1 or x_r2 == index:
            x_r2 = random.randint(0, Lambda + len(A) - 1)
        X = []
        for i in range(D): 
            X.append(<double>(Q[index].x[i] + f * (_Q[0].x[i] - Q[index].x[i]) + f * (Q[x_r1].x[i] - QA[x_r2].x[i])))
            if X[i] < self.lb[i]:
                X[i] = (self.lb[i] + X[i]) / 2
            elif X[i] > self.ub[i]:
                X[i] = (self.ub[i] + X[i]) / 2
        QA.clear()
        return X


    cdef list mutation_2(self, list Q, int Lambda, int D, double f, double grate, int jugg):
        cdef:
            int x_1, x_2, x_3, i
            list X, _Q
        x_1, x_2, x_3 = random.sample(range(Lambda), 3)
        _Q = Q[:]
#        if jugg == -1 :
#            if _Q[x_1].vio < 10.0**-100 and _Q[x_2].vio < 10.0**-100 and _Q[x_1].obj > _Q[x_2].obj: 
#                x_1, x_2 = x_2, x_1
#            elif _Q[x_1].vio > _Q[x_2].vio:
#                x_1, x_2 = x_2, x_1
#            if _Q[x_1].vio < 10.0**-100 and _Q[x_3].vio < 10.0**-100 and _Q[x_1].obj > _Q[x_3].obj: 
#                x_1, x_3 = x_3, x_1
#            elif _Q[x_1].vio > _Q[x_3].vio:
#                x_1, x_3 = x_3, x_1
            
#        if jugg != -1 or grate < 0.4:
#            if _Q[x_1].obj > _Q[x_2].obj: 
#                x_1, x_2 = x_2, x_1
#            if _Q[x_1].obj > _Q[x_3].obj: 
#                x_1, x_3 = x_3, x_1
    
#        else:
#            if _Q[x_1].vio < 10.0**-100 and _Q[x_2].vio < 10.0**-100 and _Q[x_1].obj > _Q[x_2].obj: 
#                x_1, x_2 = x_2, x_1
#            elif _Q[x_1].vio > _Q[x_2].vio:
#                x_1, x_2 = x_2, x_1
#            if _Q[x_1].vio < 10.0**-100 and _Q[x_3].vio < 10.0**-100 and _Q[x_1].obj > _Q[x_3].obj: 
#                x_1, x_3 = x_3, x_1
#            elif _Q[x_1].vio > _Q[x_3].vio:
#                x_1, x_3 = x_3, x_1
#                
#            if _Q[x_1].obj > _Q[x_2].obj: 
#                x_1, x_2 = x_2, x_1
#            if _Q[x_1].obj > _Q[x_3].obj: 
#                x_1, x_3 = x_3, x_1
                
#        if _Q[x_1].obj > _Q[x_2].obj: 
#            x_1, x_2 = x_2, x_1
#        if _Q[x_1].obj > _Q[x_3].obj: 
#            x_1, x_3 = x_3, x_1
            
        X = []
        for i in range(D): 
            X.append(<double>(_Q[x_1].x[i] + f * (_Q[x_2].x[i] - _Q[x_3].x[i])))
            if X[i] < self.lb[i]:
                X[i] = (self.lb[i] + X[i]) / 2
            elif X[i] > self.ub[i]:
                X[i] = (self.ub[i] + X[i]) / 2
        return X
    
    
                
        
    cdef void crossover_1(self, list QParent, list C_X, int D, double cr, int idx):
        cdef:
            int jRand, j
        jRand = random.randint(0, D - 1)
        for j in range(D):
            if jRand != j and random.random() <= cr:
                C_X[j] = QParent[idx].x[j]
                    
    cdef void crossover_2(self, list QParent, list C_X, int D, double cr, int idx):
        cdef:
            int n, L, j
        n = random.randint(0, D - 1)
        L = 0
        while random.uniform(0,1) <= cr and L < D - 1:
            C_X[(n + L) % D] = QParent[idx].x[(n + L) % D]
            L += 1  
    
#    cdef void crossover_2(self, list P, list C_X, int D, double cr, int idx):
#        cdef:
#            int L, j
#        L = random.randint(1, D - 1)
#        while random.uniform(0,1) <= cr and L < D:
#            C_X[L] = P[idx].x[L]
#            L += 1
            
    cdef list DE(self, list P, list A, int NP, int D, double cr, double f, int Stg, int index, double grate, int jugg):
        cdef:
            list C_X
        C_X = []
        if Stg == 0:
            C_X = self.mutation_1(P, A, NP, D, f, index, grate, jugg)
            self.crossover_1(P, C_X, D, cr, index)
        elif Stg == 1:
            C_X = self.mutation_1(P, A, NP, D, f, index, grate, jugg)
            self.crossover_2(P, C_X, D, cr, index)
        elif Stg == 2:
            C_X = self.mutation_2(P, NP, D, f, grate, jugg)
            self.crossover_1(P, C_X, D, cr, index)
        elif Stg == 3:
            C_X = self.mutation_2(P, NP, D, f, grate, jugg)
            self.crossover_2(P, C_X, D, cr, index)
        return C_X
    
   
    
    cdef int feaRule(self, list Q, int Lambda):
        cdef:
            int jugg, i
        jugg = -1   #no feasible
        for i in range(Lambda):
            if Q[i].vio < 10.0**-100:
                jugg = 1
                break
        if jugg == 1:
            for i in range(Lambda):
                if Q[i].vio > 10.0**-100:
                    jugg = 0
                    break
        return jugg
      
        
    cdef list Eq_max_min(self, list P, int NP, int bestIndex):
        cdef:
            list Q_Eq
            int i
        Q_Eq = []
        for i in range(NP):
#            Q_Eq.append(abs(sum(c**2 for c in P[bestIndex].x) - sum(c**2 for c in P[i].x)))
            Q_Eq.append(abs(P[bestIndex].obj - P[i].obj))
#        Q_Eq.sort()
        return [<double>max(Q_Eq), <double>min(Q_Eq)]
#        return [<double>max(Q_Eq[:<int>(NP/2)]), <double>min(Q_Eq[:<int>(NP/2)])] 
    
        
    cdef double Eq(self, list P, ind, double eq_max, double eq_min, int bestIndex):
        return (<double>(abs(P[bestIndex].obj - ind.obj) - eq_min) / (eq_max - eq_min + 10.0**-100))

    
    cdef list H1_max_min(self, list P, int NP):
        cdef:
            list Q_H1
            int i
        Q_H1 = []
        for i in range(NP):
            Q_H1.append(P[i].obj) 
#        Q_H1.sort()
        return [<double>max(Q_H1), <double>min(Q_H1)]        
#        return [<double>max(Q_H1[:<int>(NP/2)]), <double>min(Q_H1[:<int>(NP/2)])]        
    
    cdef double H1(self, ind, double h1_max, double h1_min):
        return <double>((ind.obj - h1_min) / (h1_max - h1_min + 10.0**-100))
    
    cdef list H2_max_min(self, list P, int NP):
        cdef:
            list Q_H2
            int i
        Q_H2 = []
        for i in range(NP):
            Q_H2.append(P[i].vio)
#        Q_H2.sort()
        return [max(Q_H2), min(Q_H2)]   
#        return [max(Q_H2[:<int>(NP/2)]), min(Q_H2[:<int>(NP/2)])]        
    
    cdef double H2(self, ind, double h2_max, double h2_min):
        return <double>((ind.vio - h2_min) / (h2_max - h2_min + 10.0**-100))
        
    


#    cdef double F_i(self, double eq, double h1, double h2, int Lambda, int idx, double para, int jugg):
#        cdef:
#            double w_i
#   
#        w_i = <double>1 / (1 + e**(-(para - 0.4) * 2)) * (<double>(idx + 1) / Lambda)**2
#        return para**100 * (<double>(idx + 1) / Lambda) * eq + (1 - w_i) * (h1 )   +  w_i * (h2)
        
    cdef double F_i(self, double eq, double h1, double h2, int Lambda, int idx, double para, int jugg):
        cdef:
            double w_i, w_t
   
#        w_i = <double>1 / (1 + e**(-(para - 0.4) * 2)) * (<double>(idx + 1) / Lambda)**2
        w_t = para 
        w_i = (<double>idx + 1)/ Lambda 
#        return para**(20 * (idx+ 1)) * eq + (1 - w_i) * (h1 ) * (0.5 + <double>(idx + 1) / Lambda)   +  w_i * (h2) 
#        return para**(20 * (idx+ 1)) * eq + (1 - w_t) * (h1 ) ** w_i  +  w_t * (h2) ** w_i
        return para**(20 * (idx+ 1)) * eq + 1.0 * (1 - w_i) * (1 - w_t)** (w_i * 5.0)  * (h1 )  +  w_i * w_t** (w_i * 5.0)  * (h2)
#        return w_i * eq + (1 - w_i) * (h1 )  +  w_i * (h2)
        
    cdef void selection(self, int i, int stg, list QParent, list QChild, gbest, int Lambda, int D, double para, list A, int ASize, list CR, list F, list Success_cr, list Success_F, list fit_improve, list Num_Success_n, int jugg, list ql, double Gamma):
        cdef:
            int bestIndex, num_stg
            list _QParent, QSum, eq_max_min, h1_max_min, h2_max_min
            double eq_max, eq_min, h1_max, h1_min, h2_max, h2_min, eq_p, eq_c, h1_p, h1_c, h2_p, h2_c
        num_stg = self.params['num_stg']
        _QParent = QParent[:]
        _QParent.append(QChild[i])
        QSum = _QParent[:]
        bestIndex = self.findBest(QSum, Lambda + 1)
        if QSum[bestIndex].vio < gbest.vio:
            gbest = QSum[bestIndex]
        elif QSum[bestIndex].vio == gbest.vio and QSum[bestIndex].obj < gbest.obj:
            gbest = QSum[bestIndex]

        eq_max_min = self.Eq_max_min(QSum, 1 + Lambda, bestIndex)
        h1_max_min = self.H1_max_min(QSum, 1 + Lambda)
        h2_max_min = self.H2_max_min(QSum, 1 + Lambda)
        eq_max = eq_max_min[0]
        eq_min = eq_max_min[1]
        h1_max = h1_max_min[0]
        h1_min = h1_max_min[1]
        h2_max = h2_max_min[0]
        h2_min = h2_max_min[1]


                
        if i < Lambda:
            eq_p = self.Eq(QSum, QParent[i], eq_max, eq_min, bestIndex)
            h1_p = self.H1(QParent[i], h1_max, h1_min)
            h2_p = self.H2(QParent[i], h2_max, h2_min)
            f1_p = self.F_i(eq_p, h1_p, h2_p, Lambda, i, para, jugg)
            eq_c = self.Eq(QSum, QChild[i], eq_max, eq_min, bestIndex)
            h1_c = self.H1(QChild[i], h1_max, h1_min)
            h2_c = self.H2(QChild[i], h2_max, h2_min)
            f1_c = self.F_i(eq_c, h1_c, h2_c, Lambda, i, para, jugg)

            if f1_p > f1_c:  
                Success_cr[stg].append(CR[len(CR) - 1])
                Success_F[stg].append(F[len(F) - 1])
                Num_Success_n[stg] += 1
#                if jugg == -1 and para >= 0.4:
#                    fit_improve[stg].append(QParent[i].vio - QChild[i].vio)
#                else:
#                    fit_improve[stg].append(QParent[i].obj - QChild[i].obj)
                fit_improve[stg].append(f1_c - f1_p)
                A.append(QParent[i])
                if QParent[i].vio != gbest.vio or QParent[i].obj != gbest.obj:
#                    print(1, gbest.obj, QParent[i].obj, QChild[i].obj)
                    QParent[i] = QChild[i]
            
        _QParent.clear()
            
              


            
    cdef void stableA(self, list A, int ASize):
        if len(A) > ASize:
            for i in range(len(A) - ASize):
                A.remove(A[random.randint(0, len(A) - 1)])  
                
    cdef void UpdateMemory(self, list Memory_cr, list Memory_F, list Success_cr, list Success_F, list fit_improve, int H, list pos):
        cdef:
            int n, k, num_Scr, num_SF, num_stg
            double f1, f3, f4, weight_1, weight_2, meanScr, meanSF
            
        num_stg = self.params['num_stg']
        for k in range(num_stg):
            if Success_cr[k] != [] and Success_F[k] != []:
                num_Scr = len(Success_cr[k])
                num_SF = len(Success_F[k])
                meanScr = 0.0
                meanSF = 0.0
                weight_1 = 0.0
                f1 = 0.0
                for i in range(num_Scr):
                    weight_1 += abs(fit_improve[k][i])
                for i in range(num_Scr):
                    f1 += abs(fit_improve[k][i]) / (weight_1 + 10.0**-100) * (Success_cr[k][i])
                meanScr = f1 
#                Memory_cr[k][pos[k]] = (meanScr + Success_cr[k][num_Scr - 1]) / 2
                Memory_cr[k][pos[k]] = meanScr
                
                
                weight_2 = 0.0
                f3 = 0.0
                f4 = 0.0
                for i in range(num_SF):
                    weight_2 += abs(fit_improve[k][i])
                for i in range(num_SF):
                    f3 += abs(fit_improve[k][i]) / (weight_2 + 10.0**-100) * np.power(Success_F[k][i], 2)
                    f4 += abs(fit_improve[k][i]) / (weight_2 + 10.0**-100) * Success_F[k][i]
                meanSF = f3 / (f4 + 10.0**-100)
#                Memory_F[k][pos[k]] = (meanSF + Success_F[k][num_SF - 1]) / 2
                Memory_F[k][pos[k]] = meanSF
                
                pos[k] = pos[k] + 1
                if pos[k] > H - 1:
                    pos[k] = 0
            
    cdef int findBest(self, list P, int NP):
        cdef:
            int bestIndex, i
        bestIndex = 0
        for i in range(NP):
            if P[i].vio < P[bestIndex].vio:
                bestIndex = i
            elif P[i].vio == P[bestIndex].vio and P[i].obj < P[bestIndex].obj:
                bestIndex = i
        return bestIndex
             
#    cdef list cal_div(self, list P, int NP):
#        cdef:
#            int i
#            list Obj, Vio
#            double min_obj, max_obj, min_vio, max_vio, qua1, qua2
#        Obj, Vio = [], []
#        [Obj.append(P[i].obj) for i in range(NP)]
#        [Vio.append(P[i].vio) for i in range(NP)]
#        min_obj, max_obj = min(Obj), max(Obj)
#        min_vio, max_vio = min(Vio), max(Vio)
#        
#        for i in range(NP):
#            qua1 = (P[i].obj - min_obj) / (abs(max_obj - min_obj) + <double>10.0**-10)
#            qua2 = (P[i].vio - min_vio) / (abs(max_vio - min_vio) + <double>10.0**-10)
#            P[i].qua = qua1 + qua2
#            
#            
#        _P = P[:]
#        _P = sorted(_P, key = operator.attrgetter('qua'))
#        _P[0].cwd = 2
#        _P[NP - 1].cwd = 2
#        for i in range(1, NP - 1):
#            _P[i].cwd = _P[i + 1].qua - _P[i - 1].qua
#        _P = sorted(_P, key = operator.attrgetter('cwd'))
##        print(_P[1].cwd)
#        return _P
            
    cdef _optimize(self, int benchID, int D):
        cdef:
            int Stg, NP, NPinit, NPlast, NPmin, FES_MAX, FES, genCount, H, Lambda, ASize, bestIndex, jugg, i, j
            double para, cr, f
            list P, A, M_CR, M_F, CR, F, Num_Success_n, pos, ql, num_temp, pos_temp, ql_temp, S_CR, S_F, fit_improve
            list cr_temp, f_temp, fit_temp, QParent, QChild, QChild_X
        NPinit = 12 * D
        NP = NPinit
        FES_MAX = self.params['FES_MAX']
        H = self.params['H']
        num_stg = self.params['num_stg']
        Lambda = 12
        NPmin = Lambda 
        Gamma = 1.0 / 3.0
        P = self.initPop(NP, benchID, D)
        A = []
        Init_M = self.initMemory()
        M_CR = Init_M[0]
        M_F = Init_M[1]
        CR = []
        F = []
        Num_Success_n = []
        pos = []
        ql = []
        for i in range(num_stg):
            Num_Success_n.append(0)
            pos.append(0)
            ql.append(0.25)
        
        
        
        FES = 0
        genCount = 1
        
        
        while FES < FES_MAX * D:
            ASize = round(4.0 * NP)
            S_CR = []
            S_F = []
            fit_improve = []
            for i in range(num_stg):
                S_CR.append([])
                S_F.append([])
                fit_improve.append([])
                
             
            bestIndex = self.findBest(P, NP)
            gbest = P[bestIndex] 
            QParent = self.initQ(P, NP, Lambda)
            jugg = self.feaRule(QParent, Lambda)
            para = <double>FES / (FES_MAX * D)
            
            QChild = []
            QChild_X = []

            for idx in range(Lambda):
                Stg = self.chooseStrategy(ql, Num_Success_n)
#                Stg = self.chooseStrategy(jugg, ql, Num_Success_n)
                f_cr = self.generate_F_CR(Stg, M_CR, M_F, S_CR, CR, F, para)
                cr = f_cr[0]
                f = f_cr[1]
                QChild_X.append(self.DE(QParent, A, Lambda, D, cr, f, Stg, idx, para, jugg))
                QChild.append(Individual(benchID, D, QChild_X[idx], self.o, self.M, self.M1, self.M2))

                self.selection(idx, Stg, QParent, QChild, gbest, Lambda, D, para, A, ASize, CR, F, S_CR, S_F, fit_improve, Num_Success_n, jugg, ql, Gamma)
                self.stableA(A, ASize)

            for i in range(Lambda):
                P.append(QParent[i])
            self.UpdateMemory(M_CR, M_F, S_CR, S_F, fit_improve, H, pos)
            FES += Lambda
            genCount += 1
            
            bestIndex = self.findBest(P, NP)
            gbest = P[bestIndex]
#            
            NPlast = len(P)
#            _P = self.cal_div(P, NP)
            _P = P[:]
            if NP > Lambda:
                NP = round(<double>(NPmin - NPinit) / (FES_MAX * D) * FES + NPinit) 
            if NP < NPlast and NP >= Lambda:
#                _P = sorted(_P, key=operator.attrgetter('cwd')) 
#                for i in range(NPlast - NP):
#                    if P[i] != gbest:
#                        P.remove(_P[i])
                
                for i in range(NPlast - NP):
                    r = random.randint(0, len(P) - 1)
                    while P[r] == gbest:
                        r = random.randint(0, len(P) - 1)
                    P.remove(P[r])
                
#                _P = sorted(_P, key=operator.attrgetter('obj')) 
#                _P = sorted(_P, key=operator.attrgetter('vio')) 
#                for i in range(NPlast - NP):
#                    P.remove(_P[len(_P) - i - 1])
                    

#            print(FES, gbest.obj, gbest.vio, P[NP - 1].cwd, P[0].cwd, NP, jugg)
            
            print(FES, gbest.obj, gbest.vio, P[NP - 1].obj, P[NP - 1].vio, NP)
#            CR.clear()
#            F.clear()
            S_CR.clear()
            S_F.clear()
            fit_improve.clear()
            QChild.clear()
            QChild_X.clear()
        return [gbest.obj, gbest.vio, gbest.c1, gbest.c2, gbest.c3] 

    def optimize(self, int benchID, int D):
        return self._optimize(benchID, D)

            
#