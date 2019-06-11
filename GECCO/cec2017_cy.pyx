# -*- coding: utf-8 -*-
"""
Created on Thu May 31 00:17:14 2018

@author: burningxt
"""
import scipy.io as sio
from libc.math cimport sin, cos, pi, e, exp, sqrt
import numpy as np


cdef class CFunction(object):
    cdef list _loadmat(self, int benchID, int D):
        cdef:
            double[:, :] o = np.zeros((1, D), dtype = np.float64)
            double[:, :] M = np.zeros((D, D), dtype = np.float64)
            double[:, :] M1 = np.zeros((D, D), dtype = np.float64)
            double[:, :] M2 = np.zeros((D, D), dtype = np.float64)
            int i, j
            dict mat_contents
            dict M_D = {10: 'M_10', 30: 'M_30', 50: 'M_50', 100: 'M_100'}
            dict M1_D = {10: 'M1_10', 30: 'M1_30', 50: 'M1_50', 100: 'M1_100'}
            dict M2_D = {10: 'M2_10', 30: 'M2_30', 50: 'M2_50', 100: 'M2_100'}
        
        if benchID == 1:
            mat_contents = sio.loadmat('Function1.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]
            
        elif benchID == 2:
            mat_contents = sio.loadmat('Function2.mat')
            o_list = mat_contents['o'].tolist()   
            M_list = mat_contents[M_D[D]].tolist()  
            for i in range(D):
                o[0, i] = o_list[0][i]
                for j in range(D):
                    M[i, j] = M_list[i][j]
            
        elif benchID == 3:
            mat_contents = sio.loadmat('Function3.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]  
            
        elif benchID == 4:
            mat_contents = sio.loadmat('Function4.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]   
        
        elif benchID == 5:
            mat_contents = sio.loadmat('Function5.mat')
            o_list = mat_contents['o'].tolist()   
            M1_list = mat_contents[M1_D[D]].tolist()  
            M2_list = mat_contents[M2_D[D]].tolist()
            for i in range(D):
                o[0, i] = o_list[0][i]
                for j in range(D):
                    M1[i, j] = M1_list[i][j]
                    M2[i, j] = M2_list[i][j]
            
        elif benchID == 6:
            mat_contents = sio.loadmat('Function6.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]   
            
        elif benchID == 7:
            mat_contents = sio.loadmat('Function7.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]    
            
        elif benchID == 8:
            mat_contents = sio.loadmat('Function8.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]    
            
        elif benchID == 9:
            mat_contents = sio.loadmat('Function9.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]   
            
        elif benchID == 10:
            mat_contents = sio.loadmat('Function10.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]   
            
        elif benchID == 11:
            mat_contents = sio.loadmat('Function11.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]    
            
        elif benchID == 12:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]    
        
        elif benchID == 13:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]    
            
        elif benchID == 14:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]    
            
        elif benchID == 15:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]   
            
        elif benchID == 16:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]    
            
        elif benchID == 17:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]   
            
        elif benchID == 18:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]    
            
        elif benchID == 19:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]   
            
        elif benchID == 20:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist() 
            for i in range(D):
                o[0, i] = o_list[0][i]   
        
        elif benchID == 21:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist()   
            M_list = mat_contents[M_D[D]].tolist()  
            for i in range(D):
                o[0, i] = o_list[0][i]
                for j in range(D):
                    M[i, j] = M_list[i][j]   
            
        elif benchID == 22:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist()   
            M_list = mat_contents[M_D[D]].tolist()  
            for i in range(D):
                o[0, i] = o_list[0][i]
                for j in range(D):
                    M[i, j] = M_list[i][j]   
            
        elif benchID == 23:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist()   
            M_list = mat_contents[M_D[D]].tolist()  
            for i in range(D):
                o[0, i] = o_list[0][i]
                for j in range(D):
                    M[i, j] = M_list[i][j]   
            
        elif benchID == 24:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist()   
            M_list = mat_contents[M_D[D]].tolist()  
            for i in range(D):
                o[0, i] = o_list[0][i]
                for j in range(D):
                    M[i, j] = M_list[i][j]  
        
        elif benchID == 25:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist()   
            M_list = mat_contents[M_D[D]].tolist()  
            for i in range(D):
                o[0, i] = o_list[0][i]
                for j in range(D):
                    M[i, j] = M_list[i][j]
            
        elif benchID == 26:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist()   
            M_list = mat_contents[M_D[D]].tolist()  
            for i in range(D):
                o[0, i] = o_list[0][i]
                for j in range(D):
                    M[i, j] = M_list[i][j]  
            
        elif benchID == 27:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist()   
            M_list = mat_contents[M_D[D]].tolist()  
            for i in range(D):
                o[0, i] = o_list[0][i]
                for j in range(D):
                    M[i, j] = M_list[i][j]   
            
        elif benchID == 28:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o_list = mat_contents['o'].tolist()   
            M_list = mat_contents[M_D[D]].tolist()  
            for i in range(D):
                o[0, i] = o_list[0][i]
                for j in range(D):
                    M[i, j] = M_list[i][j]   
        return [o, M, M1, M2]
    
    def loadmat(self, benchID, D):
        return self._loadmat(benchID, D)
    
    cdef int sgn(self, double v): 
      if v < 0:
          return -1
      if v > 0:
          return 1
      return 0

    
    cdef _evaluate(self, int benchID, int D, list x, double[:, :] o, double[:, :] M, double[:, :] M1, double[:, :] M2):        
        cdef:
            double[:] z = np.zeros(D, dtype = np.float64)
            double[:] y = np.zeros(D, dtype = np.float64)
            double[:] w = np.zeros(D, dtype = np.float64)
            double[:] g = np.zeros(10, dtype = np.float64)
            double[:] h = np.zeros(10, dtype = np.float64)
            double[:] absZ = np.zeros(D, dtype = np.float64)
            double f = 0.0
            double f0 = 0.0
            double f1 = 0.0
            double g0 = 0.0
            double g1 = 0.0
            double g2 = 0.0
            double h0 = 0.0
            double h1 = 0.0
            double h2 = 0.0
            double h3 = 0.0
            double h4 = 0.0
            double h5 = 0.0
            int len_g = 0
            int len_h = 0
            int i
            int j
        if benchID == 1:   
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f0 = 0.0
                for j in range(i + 1):
                    f0 += z[j]
                f += f0**2
            for i in range(D):
                g0 += z[i]**2 - 5000 * cos(0.1 * pi * z[i]) - 4000
            g[0] = g0   
            len_g = 1
            
        elif benchID == 2:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    y[i] += z[j] * M[i, j]
            for i in range(D):
                f0 = 0.0
                for j in range(i + 1):
                    f0 += z[j]
                f += f0**2
            for i in range(D):
                g0 += y[i]**2 - 5000 * cos(0.1 * pi * y[i]) - 4000
            g[0] = g0
            len_g = 1
        
        elif benchID == 3:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f0 = 0.0
                for j in range(i + 1):
                    f0 += z[j]
                f += f0**2
            for i in range(D):
                g0 += z[i]**2 - 5000 * cos(0.1 * pi * z[i]) - 4000
            for i in range(D):
                h0 += z[i] * sin(0.1 * pi * z[i])
            g[0] = g0
            h[0] = h0
            len_g = 1
            len_h = 1
        
        elif benchID == 4:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += z[i]**2 - 10 * cos(2 * pi * z[i]) + 10
            for i in range(D):
                g0 += z[i] * sin(2 * z[i])
            for i in range(D):
                g1 += z[i] * sin(z[i])
            g[0] = -g0 
            g[1] = g1
            len_g = 2
            
        elif benchID == 5:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D - 1):
                f += 100 * (z[i]**2 - z[i+1])**2 + (z[i] - 1)**2
            for i in range(D):
                for j in range(D):
                    y[i] += z[j] * M1[i, j]
                    w[i] += z[j] * M2[i, j]
            for i in range(D):
                g0 += y[i]**2 - 50 * cos(2 * pi * y[i]) - 40
                g1 += w[i]**2 - 50 * cos(2 * pi * w[i]) - 40
            g[0] = g0
            g[1] = g1
            len_g = 2
                
        elif benchID == 6:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += z[i]**2 - 10 * cos(2 * pi * z[i]) + 10
            for i in range(D):
                h0 += -z[i] * sin(z[i])
            for i in range(D):
                h1 += z[i] * sin(pi * z[i])
            for i in range(D):
                h2 += - z[i] * cos(z[i])
            for i in range(D):
                h3 += z[i] * cos(pi * z[i])
            for i in range(D):
                h4 += z[i] * sin(2 * sqrt(abs(z[i])))
            for i in range(D):
                h5 += - z[i] * sin(2 * sqrt(abs(z[i])))
            h[0] = h0
            h[1] = h1
            h[2] = h2
            h[3] = h3
            h[4] = h4
            h[4] = h5
            len_h = 6
        
        elif benchID == 7:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += z[i] * sin(z[i])
            for i in range(D):
                h0 += z[i] - 100 * cos(0.5 * z[i]) + 100
            h1 = -h0
            h[0] = h0
            h[1] = h1
            len_h = 2
            
        elif benchID == 8:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            f = np.amax(z[:D])
            for i in range(int(D / 2)):
                y[i] = z[2 * i]
                w[i] = z[2 * i + 1]
            for i in range(int(D/2)):
                temp0 = 0
                for j in range(i + 1):
                    temp0 += y[j]
                h0 += temp0**2
            for i in range(int(D/2)):
                temp1 = 0
                for j in range(i + 1):
                    temp1 += w[j]
                h1 += temp1**2
            h[0] = h0
            h[1] = h1
            len_h = 2
            
        elif benchID == 9:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            f = np.amax(z[:D])
            for i in range(int(D / 2)):
                y[i] = z[2 * i]
                w[i] = z[2 * i + 1] 
            g0 = 1.0
            for i in range(int(D/2)):
                g0 = g0 * w[i]
            for i in range(int(D/2) - 1):
                h0 += (y[i]**2 - y[i + 1])**2
            g[0] = g0
            h[0] = h0
            len_g = 1
            len_h = 1
                
        elif benchID == 10:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            f = np.amax(z[:D]) 
            for i in range(D):
                temp0 = 0
                for j in range(i + 1):
                    temp0 += z[i]
                h0 += temp0**2
            for i in range(D - 1):
                h1 += (z[i] - z[i + 1])**2
            h[0] = h0
            h[1] = h1  
            len_h = 2
            
        elif benchID == 11:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += z[i]
            g0 = 1.0
            for i in range(D):
                g0 = g0 * z[i]
            for i in range(D - 1):
                h0 += (z[i] - z[i + 1])**2
            g[0] = g0
            h[0] = h0
            len_g = 1
            len_h = 1
        
        elif benchID == 12:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += z[i]**2 - 10 * cos(2 * pi * z[i]) + 10
            for i in range(D):
                g0 += abs(z[i])
            for i in range(D): 
                g1 += z[i]**2
            g[0] = 4 - g0
            g[1] = g1 - 4
            len_g = 2
         
        elif benchID == 13:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D - 1):
                f += (100* (z[i]**2 - z[i+1])**2 + (z[i] - 1)**2)
            for i in range(D):
                g0 += z[i]**2 - 10 * cos(2 * pi * z[i]) + 10
                g1 += z[i]
            g[0] = g0 - 100
            g[1] = g1 - 2 * D
            g[2] = 5 - g1
            len_g = 3
        
        elif benchID == 14:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f0 += z[i]**2
                f1 += cos(2 * pi * z[i])
            f = -20 * exp(- 0.2 * sqrt(1.0 / <double>D * f0)) + 20 - exp(1.0 / <double>D * f1) + e
            for i in range(1, D):
                g0 += z[i]**2
            for i in range(D):
                h0 += z[i]**2
            g[0] = g0 + 1 - abs(z[0])
            h[0] = h0 - 4
            len_g = 1
            len_h = 1
            
        elif benchID == 15:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                absZ[i] = abs(z[i])
            f = np.amax(absZ[:D])
            g0 = 0
            for i in range(D):
                g0 += z[i]**2
            g[0] = g0 - 100 * D
            h[0] = cos(f) + sin(f)
            len_g = 1
            len_h = 1
            
        elif benchID == 16:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += abs(z[i])
            for i in range(D):
                g0 += z[i]**2
            g[0] = g0 - 100 *D
            h[0] = (cos(f) + sin(f))**2 - exp(cos(f) + sin(f)) - 1 + exp(1)
            len_g = 1
            len_h = 1
            
        elif benchID == 17:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            f1 = 1.0
            for i in range(D):
                f0 += z[i]**2
                f1 = f1 * cos(z[i] / sqrt(i + 1))
            f = 1.0 / 4000.0 * f0 + 1 - f1
            for i in range(D):
                g0 = 0.0
                for j in range(D): 
                    if(j == i):
                        g0 = f0 - z[j]**2
                g1 += self.sgn(abs(z[i]) - g0 - 1)
            h0 = f0
            g[0] = 1 - g1
            h[0] = h0 - 4 * D
            len_g = 1
            len_h = 1
         
        elif benchID == 18:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                if abs(y[i]) < 0.5:
                    z[i] = y[i]
                else:
                    z[i] = 0.5 * round( 2 * y[i])
            for i in range(D):
                f += z[i]**2 - 10 * cos(2 * pi * z[i]) + 10
            for i in range(D):
                g0 += abs(y[i])
            for i in range(D):
                g1 += y[i]**2
            h1 = 1.0
            for i in range(D - 1):
                h0 += 100 * (y[i]**2 - y[i + 1])**2
                h1 = h1 * (sin(y[i] - 1))**2 * pi
            g[0] = 1 - g0
            g[1] = g1 - 100 * D
            h[0] = h0 + h1
            len_g = 2
            len_h = 1
         
        elif benchID == 19:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += (abs(z[i]))**0.5 + 2 * sin(z[i]**3)
            for i in range(D - 1):
                g0 += -10 * exp(- 0.2 * sqrt(z[i]**2 + z[i + 1]**2))
            for i in range(D):
                g1 += (sin(2 * z[i]))**2
            g[0] = g0 + (D - 1) * 10 / <double>exp(-5)
            g[1] = g1 - 0.5 * D
            len_g = 2
            
        elif benchID == 20:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D - 1):
                f += 0.5 + ((sin(sqrt(z[i]**2 + z[i + 1]**2)))**2 - 0.5) / (1 + 0.001 * sqrt(z[i]**2 + z[i+1]**2))**2
            f += 0.5 + ((sin(sqrt(z[D - 1]**2 + z[0]**2)))**2 - 0.5) / (1 + 0.001 * sqrt(z[D - 1]**2 + z[0]**2))**2
            for i in range(D):
                g0 += z[i]
            g[0] = (cos(g0))**2 - 0.25 * cos(g0) - 0.125
            g[1] = exp(cos(g0)) - exp(0.25)
            len_g = 2
#            
        elif benchID == 21:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D):
                f += z[i] **2 - 10 * cos(2 * pi * z[i]) + 10
            for i in range(D):
                g0 += abs(z[i])
                g1 += z[i]**2
            g[0] = 4 - g0
            g[1] = g1 - 4
            len_g = 2
#            
        elif benchID == 22:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D - 1):
                f += 100 * (z[i]**2 - z[i + 1])**2 + (z[i] - 1)**2
            for i in range(D):
                g0 += z[i]**2 - 10 * cos(2 * pi * z[i]) +10
                g1 += z[i]
            g[0] = g0 - 100
            g[1] = g1 - 2 * D
            g[2] = 5 - g1
            len_g = 3
                
        elif benchID== 23:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D):
                f0 += z[i]**2
                f1 += cos(2 * pi * z[i])
            f += -20 * exp(-0.2 * sqrt(1.0 / <double>D * f0)) + 20 - exp(1.0 / <double>D * f1) + e
            for i in range(1, D):
                g0 += z[i]**2
            for i in range(D):
                h0 += z[i]**2
            g[0] = g0 + 1 - abs(z[0])
            h[0] = h0 - 4
            len_g = 1
            len_h = 1
        
        elif benchID == 24:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D):
                absZ[i] = abs(z[i])
            f = np.amax(absZ[:D])
            for i in range(D):
                g0 += z[i]**2
            g[0] = g0 - 100 * D
            h[0] = cos(f) + sin(f)
            len_g = 1
            len_h = 1
            
        if benchID == 25:  
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D):
                f += abs(z[i])
            for i in range(D):
                g0 += z[i]**2
            g[0] = g0 - 100 * D
            h[0] = (cos(f) + sin(f))**2 - exp(cos(f) + sin(f)) - 1 + exp(1)
            len_g = 1
            len_h = 1
        
        elif benchID == 26:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            f1 = 1.0
            for i in range(D):
                f0 += z[i]**2
                f1 = f1 * cos(z[i] / sqrt(i + 1))
            f = 1.0 / 4000.0 * f0 + 1 - f1
            for i in range(D):
                g0 = 0.0
                for j in range(D): 
                    if(j == i):
                        g0 = f0 - z[j]**2
                g1 += self.sgn(abs(z[i]) - g0 - 1)
            h0 = f0
            g[0] = 1 - g1
            h[0] = h0 - 4 * D
            len_g = 1
            len_h = 1
            
        elif benchID == 27:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            if abs(z[i]) < 0.5:
                z[i] = y[i]
            else:
                z[i] = 0.5 * round(2 * y[i])
            for i in range(D):
                f += z[i]**2 - 10 * cos(2 * pi * z[i]) + 10
            for i in range(D):
                g0 += abs(y[i])
            for i in range(D):
                g1 += y[i]**2
            for i in range(D - 1):
                h0 += 100 * (y[i]**2 - y[i + 1])**2
                h1 = h1 * (sin(y[i] - 1))**2 * pi
            g[0] = 1 - g0
            g[1] = g1 - 100 * D
            h[0] = h0 + h1
            len_g = 2
            len_h = 1
            
        elif benchID == 28:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D):
                f += (abs(z[i]))**0.5 + 2 * sin(z[i]**3)
            for i in range(D - 1):
                g0 += -10 * exp(- 0.2 * sqrt(z[i]**2 + z[i + 1]**2))
            for i in range(D):
                g1 += (sin(2 * z[i]))**2
            g[0] = g0 + (D - 1) * 10 / <double>exp(-5)
            g[1] = g1 - 0.5 * D
            len_g = 2
            
            
 
        cdef int c1 = 0 
        cdef int c2 = 0
        cdef int c3 = 0
        cdef double v = 0.0
        if g != []:
            for i in range(len_g):
                if max(0.0, g[i]) > 1:
                    c1 += 1
                if max(0.0, g[i]) > 0.01 and max(0.0, g[i]) < 1:
                    c2 += 1
                if max(0.0, g[i]) > 0.0001 and max(0.0, g[i]) < 0.01:
                    c3 += 1
            for i in range(len_g):
                v += max(0.0, g[i])
        if h != []:
            for i in range(len_h):
                if max(0.0, abs(h[i]) - 0.0001) > 1:
                    c1 += 1
                if max(0.0, abs(h[i]) - 0.0001) > 0.01 and max(0.0, abs(h[i]) - 0.0001) < 1:
                    c2 += 1
                if max(0.0, abs(h[i]) - 0.0001) > 0.0001 and max(0.0, abs(h[i]) - 0.0001) < 0.01:
                    c3 += 1
            for i in range(len_h):
                v += max(0.0, abs(h[i]) - 0.0001)
        v = v / (len_g + len_h)
        return [f, v, c1, c2, c3] #objevtive value and violation degree
    
    def evaluate(self, int benchID, int D, list x, double[:, :] o, double[:, :] M, double[:, :] M1, double[:, :] M2): 
        return self._evaluate(benchID, D, x, o, M, M1, M2)
               
    cdef list _getLowBound(self, int benchID, int D):
        cdef:
            list lb
            int i
        lb = []
        if benchID == 1:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 2:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 3:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 4:
            for i in range(D):
                lb.append(-10.0)
        elif benchID == 5:
            for i in range(D):
                lb.append(-10.0)
        elif benchID == 6:
            for i in range(D):
                lb.append(-20.0)    
        elif benchID == 7:
            for i in range(D):
                lb.append(-50.0)
        elif benchID == 8:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 9:
            for i in range(D):
                lb.append(-10.0) 
        elif benchID == 10:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 11:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 12:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 13:
            for i in range(D):
                lb.append(-100.0)       
        elif benchID == 14:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 15:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 16:
            for i in range(D):
                lb.append(-100.0)  
        elif benchID == 17:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 18:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 19:
            for i in range(D):
                lb.append(-50.0)
        elif benchID == 20:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 21:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 22:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 23:
            for i in range(D):
                lb.append(-100.0)       
        elif benchID == 24:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 25:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 26:
            for i in range(D):
                lb.append(-100.0)  
        elif benchID == 27:
            for i in range(D):
                lb.append(-100.0)
        elif benchID == 28:
            for i in range(D):
                lb.append(-50.0)
        return lb
    
    def getLowBound(self, benchID, D):
        return self._getLowBound(benchID, D)
    
    cdef list _getUpBound(self, int benchID, int D):
        cdef:
            list ub
            int i
        ub = []
        if benchID == 1:
            for i in range(D):
                ub.append(100.0)
        if benchID == 2:
            for i in range(D):
                ub.append(100.0)
        if benchID == 3:
            for i in range(D):
                ub.append(100.0)
        if benchID == 4:
            for i in range(D):
                ub.append(10.0)
        elif benchID == 5:
            for i in range(D):
                ub.append(10.0)
        elif benchID == 6:
            for i in range(D):
                ub.append(20.0)
        elif benchID == 7:
            for i in range(D):
                ub.append(50.0)
        elif benchID == 8:
            for i in range(D):
                ub.append(100.0)
        elif benchID == 9:
            for i in range(D):
                ub.append(10.0)
        elif benchID == 10:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 11:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 12:
            for i in range(D):
                ub.append(100.0)    
        elif benchID == 13:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 14:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 15:
            for i in range(D):
                ub.append(100.0) 
        elif benchID == 16:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 17:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 18:
            for i in range(D):
                ub.append(100.0) 
        elif benchID == 19:
            for i in range(D):
                ub.append(50.0)
        elif benchID == 20:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 21:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 22:
            for i in range(D):
                ub.append(100.0)    
        elif benchID == 23:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 24:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 25:
            for i in range(D):
                ub.append(100.0) 
        elif benchID == 26:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 27:
            for i in range(D):
                ub.append(100.0)       
        elif benchID == 28:
            for i in range(D):
                ub.append(50.0) 
        return ub
    
    def getUpBound(self, int benchID, int D):
        return self._getUpBound(benchID, D)
    
    def getParam(self, int benchID, int D):
        return D

cdef class Individual:
    #obj: the objective value
    #vio: the value of violation degree
    #benchID: the ID of benchmark function
    #x: the paremeters
    #c1: the number of violations by more than 1.0
    #c2: the number of violations in the range [0.01, 1.0]
    #c3: the number of violations in the range [0.0001, 0.01]
    cdef public:
        double obj
        double vio
        int c1
        int c2
        int c3
        list x
        double stg
        double qua
        double fea
        double gen
        double eqv
        double cwd
    
    def __init__(self, int benchID, int D, list X, double[:, :] o, double[:, :] M, double[:, :] M1, double[:, :] M2):
        cdef list p
        p = CFunction().evaluate(benchID, D, X, o, M, M1, M2)
        self.x = X
        self.obj = p[0]
        self.vio = p[1]
        self.c1 = p[2]
        self.c2 = p[3]
        self.c3 = p[4]
        self.stg = 0
        self.qua = 0.0
        self.fea = 0.0
        self.gen = 0.0
        self.eqv = 0.0
        self.cwd = 0.0

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        