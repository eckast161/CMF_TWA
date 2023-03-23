
import numpy as np
from numpy import cos, sin
from numpy import sqrt
import time
from scipy.sparse.linalg import expm
import pandas as pd

label_size = 20



import numpy as np
#from google.colab import files

"""In order to provide a better presentation of the graphs we use the rcParams options shown below."""


from numpy import cos, sin

label_size = 20

from numpy import sqrt

import time
from numpy import sin, cos

T0=time.time()


# In[3]:



from numpy import random, zeros, sqrt,cos,sin

import numpy as np

import numbers

from numpy import sqrt, zeros, diag, random, real, array, complex_, transpose, conjugate, exp, matmul, real



from scipy.linalg import norm
from numpy import dot

from numpy import pi as pi
from numpy import array

import scipy

from numpy import exp, imag, real,zeros, conj
from numpy import sum
from scipy import sparse
from numpy import array
import scipy.sparse as sp
from numpy import concatenate

from scipy.sparse import csr_matrix
from numpy import sqrt, dot
from numpy.random import choice
from numpy.random import normal
from numpy import sqrt


from numpy import add
from numpy import concatenate
from scipy.sparse import coo_matrix

from scipy.sparse import kron

#t0=time.time()

import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
from scipy.sparse import kron
from scipy.sparse import coo_matrix, identity



# \begin{align}
# \hat{H}_{{\rm MACE}}^{i} & =\sum_{k\in C_{i}}B_{Q}\hat{s}_{z,k}^{2}+\sum_{\substack{j,k\in C_{i},\\
# j<k
# }
# }V_{jk}\left(\hat{s}_{z,j}\hat{s}_{z,k}-\frac{1}{2}\left(\hat{s}_{x,j}\hat{s}_{x,k}+\hat{s}_{y,j}\hat{s}_{y,k}\right)\right).
# \end{align}
# 

# # Parameters

# In[4]:


#Parameters  (Check the units)

B_Q=-1.85 # in Vdd units
N_atom=192
S_spin=3   #determine the number of Zeeman levels (2S+1)
Levels=int(2*S_spin+1)
Atom_cutoff=4


#Array Vij will be provided by Sean, just we will play with a random Vij for testing that the code makes sense.
#Mgen=np.random.rand(N_atom,N_atom)


Vij_p=array(pd.read_csv('Vij.csv', header=None))


# In[5]:




import numpy as np
from numpy import sin, cos, sqrt

"""In order to provide a better presentation of the graphs we use the rcParams options shown below."""


from numpy import sqrt

import time

# In[6]:


Vij=(Vij_p-np.diag(np.diag(Vij_p)))


# In[7]:


VijN=np.zeros((N_atom,N_atom))

for j in range(N_atom):
    
    VijN[j]=Vij[j][:N_atom]
    
    


# In[8]:


Vij=VijN


# In[ ]:





# In[9]:




# In[ ]:





# In[ ]:



def index_Gij(arr,Atom_cutoff):
    
    Ordered_N = list(np.argsort(arr)[::-1][:Atom_cutoff])
    
    return Ordered_N

# Provides a list of lists with the arrays that belong to each cluster


Index_N=[]

Rem_iden=abs(Vij)


for j in range(Rem_iden.shape[0]):
    
    test_list = list(Rem_iden[j])
    
    Index_N.append( [j]+list(index_Gij(test_list,Atom_cutoff-1))  )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#https://easyspin.org/easyspin/documentation/spinoperators.html


#Define Sz

Val=np.linspace(-S_spin,S_spin,int(2*S_spin+1))
Coord=np.linspace(0,2*S_spin,int(2*S_spin+1))
    
Sz_sp=sparse.coo_matrix((Val,(Coord,Coord)),shape=(int(2*S_spin+1),int(2*S_spin+1)),dtype="complex") 


# In[ ]:


#Define Splus

Val=[]


for j in range(1,int(2*S_spin)+1):
    
    Val.append(sqrt((S_spin)*(S_spin+1)-(S_spin-j)*(S_spin+1-j)))
    

Coord=np.linspace(0,int(2*S_spin)-1,int(2*S_spin))    
Splu_sp=(sparse.coo_matrix((Val,(Coord,Coord+1)),shape=(int(2*S_spin)+1,int(2*S_spin)+1),dtype="complex")) 

Sx_sp=(Splu_sp+Splu_sp.T)/2
Sy_sp=(Splu_sp-Splu_sp.T)/(2j)


# In[ ]:





# In[ ]:





# In[ ]:


#Initial state |S,-S>
Init_st=(sparse.coo_matrix(([1],([Levels-1],[0])),shape=(int(2*S_spin)+1,1),dtype="complex")) 

#Rotated initial state

Init_st_rot=(expm(scipy.sparse.csc_matrix(-1j*(np.pi/2)*Sy_sp))@Init_st)


# In[ ]:


Id_Spin=scipy.sparse.identity(Levels, dtype='complex')

Zero_Spin=0*Id_Spin


def Zeeman_Popul(m_s):
    
    return sparse.coo_matrix(([1],([S_spin-m_s],[S_spin-m_s])),shape=(Levels,Levels),dtype="complex")


# In[ ]:


Init_st_rot.toarray()


# In[ ]:





# In[ ]:


Init_st_rot


# In[ ]:


Init_Total_rot=Init_st_rot

for j in range(Atom_cutoff-1):
    
    Init_Total_rot=kron(Init_Total_rot,Init_st_rot)
    


# In[ ]:


Init_Total_rot=coo_matrix(Init_Total_rot)  #Initial state wavefunction


# In[ ]:


Init_Total_rot.todense()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


Iden_Tensor=coo_matrix(identity((Levels)**Atom_cutoff, dtype='complex'))
Zero_Tensor=0*Iden_Tensor


# In[ ]:


import functools as ft
from functools import reduce

lst = Atom_cutoff*[Id_Spin]
XX = reduce(kron, lst)


# In[ ]:





# In[ ]:


#pip install pympler


# In[ ]:


def A_i_tensor(A_op,i_index):
    
    Array_id=Atom_cutoff*[Id_Spin] 
    
    Array_id[i_index]=A_op
    
    return reduce(kron, Array_id)
    


# In[ ]:





# In[ ]:





# In[ ]:


#Not needed, just for reference

def tensor_kjXYZ2( k , j):   #tensor among k,j are Positions in the cluster after the G classification
                         #k is +sigma and j is -sigma
        
    Array_idX   = Atom_cutoff*[Id_Spin]   
    Array_idX[k]= Sx_sp
    Array_idX[j]= Sx_sp
    
    Array_idY   = Atom_cutoff*[Id_Spin]
    Array_idY[k]= Sy_sp
    Array_idY[j]= Sy_sp
    
    Array_idZ   = Atom_cutoff*[Id_Spin]
    Array_idZ[k]= Sz_sp
    Array_idZ[j]= Sz_sp

    return reduce(kron, Array_idX),reduce(kron, Array_idY),reduce(kron, Array_idZ) 


# In[ ]:





# In[ ]:


def tensor_termIntact( k , j):   #tensor among k,j are Positions in the cluster after the G classification
                         #k is +sigma and j is -sigma
        
    Array_idX   = Atom_cutoff*[Id_Spin]   
    Array_idX[k]= Sx_sp
    Array_idX[j]= Sx_sp
    
    Array_idY   = Atom_cutoff*[Id_Spin]
    Array_idY[k]= Sy_sp
    Array_idY[j]= Sy_sp
    
    Array_idZ   = Atom_cutoff*[Id_Spin]
    Array_idZ[k]= Sz_sp
    Array_idZ[j]= Sz_sp

    return reduce(kron, Array_idZ)-0.5*( reduce(kron, Array_idX)+reduce(kron, Array_idY) ) 


# In[ ]:





# In[ ]:





# In[ ]:


#Not needed, just for reference

def tensor_kjXX( k , j):   #tensor among k,j are Positions in the cluster after the G classification
                         #k is +sigma and j is -sigma 
    Array_id=Atom_cutoff*[Id_Spin]  
    
    Array_id[k]=Sx_sp
    Array_id[j]=Sx_sp

    return reduce(kron, Array_id)



def tensor_kjYY( k , j):   #tensor among k,j are Positions in the cluster after the G classification
                         #k is +sigma and j is -sigma 
    Array_id=Atom_cutoff*[Id_Spin]  
    
    Array_id[k]=Sy_sp
    Array_id[j]=Sy_sp

    return reduce(kron, Array_id)



def tensor_kjZZ( k , j):   #tensor among k,j are Positions in the cluster after the G classification
                         #k is +sigma and j is -sigma 
    Array_id=Atom_cutoff*[Id_Spin]  
    
    Array_id[k]=Sz_sp
    Array_id[j]=Sz_sp

    return reduce(kron, Array_id)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


Sz_cuad=Sz_sp@Sz_sp


# In[ ]:


def tensor_k_Zcuad( k ):  #Excited state k is a Position in the cluster after the G classification
    
    Array_id=Atom_cutoff*[Id_Spin]  
    
    Array_id[k]=Sz_cuad
        
    return reduce(kron, Array_id)


# In[ ]:





# In[ ]:


# Construct a vector with |C_i| elements equal to A_i_tensor(Sz_cuad,j)

Sz_cuad_array=Zero_Tensor


for j in range(Atom_cutoff):
    
    Sz_cuad_array+= B_Q* A_i_tensor(Sz_cuad,j)
    


# In[ ]:





# In[ ]:





# In[ ]:





# \begin{align}
# \hat{H}_{{\rm MACE}}^{i} & =\sum_{k\in C_{i}}B_{Q}\hat{s}_{z,k}^{2}+\sum_{\substack{j,k\in C_{i},\\
# j<k
# }
# }V_{jk}\left(\hat{s}_{z,j}\hat{s}_{z,k}-\frac{1}{2}\left(\hat{s}_{x,j}\hat{s}_{x,k}+\hat{s}_{y,j}\hat{s}_{y,k}\right)\right).
# \end{align}
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def Hamilt_ic(index_center):
     
    Index_list=Index_N[index_center]
    Hamil=Sz_cuad_array
    
    
    for j in range(Atom_cutoff):
        
        IL_j   = Index_list[j]
        
        for k in range(j+1,Atom_cutoff):
            
            
            IL_k   = Index_list[k]

            Ten_jk = tensor_termIntact( j , k)

            
            Hamil=Hamil+ Vij[IL_j,IL_k]*Ten_jk
            
            #print(j,k,"IndexN", IL_j,IL_k)
            

    return  Hamil


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def f(t, yR,M_vect,b):

    return -1j*M_vect@yR


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



import numpy as np
#from google.colab import files

"""In order to provide a better presentation of the graphs we use the rcParams options shown below."""


from numpy import cos, sin

label_size = 20

from numpy import sqrt

import time
from numpy import sin, cos

T0=time.time()


# In[ ]:


Zeeman_Operators=[]


for j in range(-S_spin,S_spin+1):
    
    Zeeman_Operators.append(A_i_tensor(Zeeman_Popul( -j),0))
    


# In[ ]:


tspan=np.linspace(0,2,400)



from scipy.integrate import solve_ivp


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


T_init=tspan[0]

T_fin =tspan[-1]


# In[ ]:





# In[ ]:





# In[ ]:


from scipy.integrate import solve_ivp


def Dynam_Popul_CI(index_center):
    
    Populations=np.zeros((Levels,len(tspan)))
    

    #sol = solve_ivp(f, [tspan[0], tspan[-1]], list(Init_Total_rot.toarray().T[0]),args=(Hamilt_ic(index_center),None),dense_output=True )
    sol = solve_ivp(f, [T_init, T_fin], list(Init_Total_rot.toarray().T[0]),args=(Hamilt_ic(index_center),None),dense_output=True,max_step=0.0001 )

    Sol_dense=sol.sol(tspan)
    
    
    for j in range(Levels):
        
        Populations[j]=real((diag((conj(Sol_dense.T)@(Zeeman_Operators[j]@Sol_dense)))))
        
        
    return Populations

    


# In[ ]:





# In[ ]:





# In[ ]:


Populat_Fin=np.zeros((Levels,len(tspan)))


# In[ ]:





# In[ ]:


#Populat_Fin=Dynam_Popul_CI(0) #any number is ok for the argument


# In[ ]:





# In[ ]:





# In[ ]:


VarL=(3**2)*Populat_Fin[0]
VarL=VarL+(2**2)*Populat_Fin[1]
VarL=VarL+(1**2)*Populat_Fin[2]
VarL=VarL+(0**2)*Populat_Fin[3]
VarL=VarL+((-1)**2)*Populat_Fin[4]
VarL=VarL+((-2)**2)*Populat_Fin[5]
VarL=VarL+((-3)**2)*Populat_Fin[6]


# In[ ]:


Atom_cutoff


# In[ ]:


#CorrT2=N_atom*(S_spin/2-VarL)
#CorrT3=N_atom*(S_spin/2-VarL)
#CorrT4=N_atom*(S_spin/2-VarL)
#CorrT5=N_atom*(S_spin/2-VarL)
#CorrT6=S_spin/2-VarL


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # CMFT

# In[ ]:





# In[ ]:





# In[ ]:


#(Vij@Mean_Val_XYZ)


OperX=[]
OperY=[]
OperZ=[]


for j in range(Atom_cutoff):
    
    
    OperX.append( A_i_tensor(Sx_sp,j) )
    OperY.append( A_i_tensor(Sy_sp,j) )
    OperZ.append( A_i_tensor(Sz_sp,j) )


# In[ ]:





# \begin{align*}
# \hat{H}_{{\rm CMFT}}^{i} & =\sum_{k\in C_{i}}B_{Q}\hat{s}_{z,k}^{2}+\sum_{\substack{j,k\in C_{i},\\
# j<k
# }
# }V_{jk}\left(\hat{s}_{z,j}\hat{s}_{z,k}-\frac{1}{2}\left(\hat{s}_{x,j}\hat{s}_{x,k}+\hat{s}_{y,j}\hat{s}_{y,k}\right)\right)+\sum_{\substack{j\in C_{i},}
# }\left(\hat{s}_{z,j}\vec{V}_{j}\cdot\left\langle \vec{\hat{s}_{z}}\right\rangle -\frac{1}{2}\left(\hat{s}_{x,j}\vec{V}_{j}\cdot\left\langle \vec{\hat{s}_{x}}\right\rangle +\hat{s}_{y,j}\vec{V}_{j}\cdot\left\langle \vec{\hat{s}_{y}}\right\rangle \right)\right)\\
#  & \,\,\,\,\,-\sum_{\substack{j,k\in C_{i}}
# }V_{jk}\left(\hat{s}_{z,j}\left\langle \hat{s}_{z,k}\right\rangle -\frac{1}{2}\left(\hat{s}_{x,j}\left\langle \hat{s}_{x,k}\right\rangle +\hat{s}_{y,j}\left\langle \hat{s}_{y,k}\right\rangle \right)\right).
# \end{align*}

# In[ ]:


Init_Total_rot


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



number_parts=60


index_parts=int(len(tspan)/number_parts)

tnew=[]

for tind_ in range(len(tspan)-1):

    Time_pos=int((tind_+1)//index_parts)
    
    if (tind_+1)%index_parts==0:
        
        tnew.append(tspan[tind_+1])

tnew=array(tnew)




# In[ ]:





# In[ ]:


len(tnew)+1


# In[ ]:





# In[ ]:



def Hamilt_ic_MACE(index_center):
     
    Index_list=Index_N[index_center]
    Hamil=Sz_cuad_array
    
    
    for j in range(Atom_cutoff):
        
        IL_j   = Index_list[j]
        
        for k in range(j+1,Atom_cutoff):
            
            
            IL_k   = Index_list[k]


            
            Hamil=Hamil+ Vij[IL_j,IL_k]*(tensor_termIntact( j , k) )
            
            #print(j,k,"IndexN", IL_j,IL_k)
            

    return  Hamil


# In[ ]:





# In[ ]:





# In[ ]:


SumVij=sum(Vij,axis=0)

SumClust=SumVij



for j in range(N_atom):
    for w in Index_N[0]:
        
        SumClust[j]=SumVij[j]-Vij[j,w]
        


# In[ ]:


Index_N[0]

SumClusList=[]

for j_ in Index_N[0]:
    
    SumClusList.append(SumClust[j_])
    
    


# In[ ]:





# In[ ]:



# Wavefunction saving

Wave_function=np.zeros((Levels**Atom_cutoff,1),dtype="complex")

for j in range(1):    
    Wave_function[:,j]=Init_Total_rot.toarray().reshape(Levels**Atom_cutoff,)   #Not sparse structure because the state is not sparse
    

Mean_Val_XYZ=np.zeros((1,3),dtype="complex")

Mean_Val_XYZ[:,0]=diag(conj(Wave_function.T)@(OperX[0]@Wave_function))
Mean_Val_XYZ[:,1]=diag(conj(Wave_function.T)@(OperY[0]@Wave_function))
Mean_Val_XYZ[:,2]=diag(conj(Wave_function.T)@(OperZ[0]@Wave_function))


Popul_CMFT=np.zeros((2*S_spin+1,len(tnew)+1),dtype=complex)



for j in range(2*S_spin+1):
    Popul_CMFT[j,0]=sum(diag(conj(Wave_function.T)@(Zeeman_Operators[j]@Wave_function)))


# In[ ]:


#sum(Popul_CMFT[:,0])


# In[ ]:





# In[ ]:


def Hamilt_ic_CMFT(index_center,ArrayXYZ):
     
    Index_list=Index_N[index_center]
    Hamil=Zero_Tensor
    
    for j in range(Atom_cutoff):
        

        Xmean,Ymean,Zmean=ArrayXYZ[0]
        
        hat_Sxj=OperX[j]
        hat_Syj=OperY[j]
        hat_Szj=OperZ[j]
        
        Hamil=Hamil+ SumClusList[j]*(Zmean*hat_Szj - 0.5*( Xmean * hat_Sxj + Ymean * hat_Syj   ) )
            

    return  Hamil


# In[ ]:


Wave_function


# In[ ]:


max_step=0.1


# In[ ]:



def Wave_func_part(Wave_fun_mt,inst_time,ArrayXYZ):
    
    Result_fin=np.zeros(((Levels**Atom_cutoff),1),dtype=complex)
    
    cent_ind=0

    Mat_dyn=Hamilt_ic_MACE(cent_ind)+Hamilt_ic_CMFT(cent_ind,ArrayXYZ)
        
    sol_cent_ind = solve_ivp(f,[tspan[inst_time],tspan[inst_time+1]],list(Wave_fun_mt[:,0])
                            ,args=(Mat_dyn,None),dense_output=False,max_step=max_step)# , t_eval= [tspan[inst_time+1]])
        
    Result_fin[:,cent_ind]=((sol_cent_ind.y)[:,-1]).reshape((Levels**Atom_cutoff,))
        

        
    return Result_fin


# In[ ]:





# In[ ]:


cent_ind=0
Hamilt_ic_MACE(cent_ind)+Hamilt_ic_CMFT(cent_ind,Mean_Val_XYZ)


# In[ ]:





# In[ ]:





# In[ ]:


count=0

tNEW=[0]

for tind_ in range(len(tspan)-1):
    
    Wave_function= Wave_func_part(Wave_function,tind_,Mean_Val_XYZ)
    
    if (tind_+1)%index_parts==0:
        count=count+1
        
        tNEW.append(tspan[tind_+1])
        
        for j in range(Levels):
            Popul_CMFT[j,count]=sum(diag(conj(Wave_function.T)@(Zeeman_Operators[j]@Wave_function)))
        
        #Popul_CMFT[j,count]=sum(diag(conj(Wave_function.T)@(Zeeman_Operators[j]@Wave_function)))

        #print(100*(tind_+1)/len(tspan))
        
    print((100*tind_/(len(tspan)-1)),"%")
        
    Mean_Val_XYZ=np.zeros((1,3),dtype="complex")
    Mean_Val_XYZ[:,0]=diag(conj(Wave_function.T)@(OperX[0]@Wave_function))
    Mean_Val_XYZ[:,1]=diag(conj(Wave_function.T)@(OperY[0]@Wave_function))
    Mean_Val_XYZ[:,2]=diag(conj(Wave_function.T)@(OperZ[0]@Wave_function))


# In[ ]:


Mean_Val_XYZ


# In[ ]:





# In[ ]:


Popul_CMFT


# In[ ]:





# In[ ]:


VarLCMF=(3**2)*Popul_CMFT[0]
VarLCMF=VarLCMF+(2**2)*Popul_CMFT[1]
VarLCMF=VarLCMF+(1**2)*Popul_CMFT[2]
VarLCMF=VarLCMF+(0**2)*Popul_CMFT[3]
VarLCMF=VarLCMF+((-1)**2)*Popul_CMFT[4]
VarLCMF=VarLCMF+((-2)**2)*Popul_CMFT[5]
VarLCMF=VarLCMF+((-3)**2)*Popul_CMFT[6]


# In[ ]:


Atom_cutoff


# In[ ]:


CorrT2CMF=N_atom*(S_spin/2-VarLCMF)

#CorrT2CMA=N_atom*(S_spin/2-VarLCMF)


# In[ ]:





# In[ ]:


CorrT2CMF.shape


# In[ ]:


len(tNEW)


# In[ ]:


CorrT2CMF


# In[ ]:


Atom_cutoff


# In[ ]:








# In[ ]:





# In[ ]:


Random_signature=np.random.rand(1,1)[0,0].round(6)


# In[ ]:





# In[ ]:





# In[ ]:


VarL0dot001=VarL


# In[ ]:





# In[50]:


Shape=VarL0dot001.shape[0]


# In[51]:


VarL0dot001=VarL0dot001.reshape(Shape,1)


# In[52]:


tspan=tspan.reshape(Shape,1)


# In[ ]:


Res_conc=np.concatenate((tspan,VarL0dot001),axis=1)
Res_conc=np.concatenate((Res_conc[0].reshape(1,len(Res_conc[0])),Res_conc),axis=0)


# In[57]:




# In[58]:


file = open("_CMFT_N_atom_"+str(N_atom)+"_Cluster_size_"+str(Atom_cutoff)+"_Max_step_"+str(max_step)+"_Tspan_"+str(tspan[0,0])+"-"+str(tspan[-1,0])+"_Signature_"+str(Random_signature)+".txt", "w+")
np.savetxt(file, Res_conc, delimiter=',')
file.write('\n')
file.write('\n')
file.close()   


#import pandas as pd
#Res0128_1dot25_0=np.array(pd.read_csv('_MACE_AV__N_atom192_Cluster_size_4_Max_step_0.001_Tspan_0.0-2.0_Signature_0.520019.txt')).astype(complex)


# In[ ]:



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# In[ ]:





# In[ ]:





# In[ ]:




