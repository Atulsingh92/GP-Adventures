
#%%
import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab as pb
import GPy
from gpflow.ci_utils import ci_niter
from pykrige import OrdinaryKriging
from math import pi as pi
import pyKriging  
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan
import pyDOE as pyd
from sklearn import preprocessing
import seaborn as sb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams.update(plt.rcParamsDefault)
np.random.seed(123)
np.set_printoptions(suppress=True)

#%% x assignment
#k = 3

#n_samples = 60
#sp = samplingplan(k)  
#X = sp.optimallhc(n_samples)

#bounds1 = np.array([[-2,2]]) 
#xlim=(-2,2)
#ylim=(-3e5,8e5)
xlim=(-12,12)
ylim=(-3e5,8e5)

#bounds = np.array([[-2,2]]) 
bounds = np.array([[min(xlim),max(xlim)]]) 
bounds1 = np.array([[min(xlim),max(xlim)]])

X1 = pyd.lhs(3,10, criterion='centermaximin') # 3 cols of 60 samples
X2= X1*(np.max(bounds1)-np.min(bounds1))+np.min(bounds1)

#training set
x1_train = X2[:,0]
x2_train = X2[:,1]
x3_train = X2[:,2]

#testing set
n_testing_samples = 20
x1_test = np.linspace(0,1, n_testing_samples)
x2_test = np.linspace(0,1, n_testing_samples)
x3_test = np.linspace(0,1, n_testing_samples)

n=50
x2 = np.linspace(0,1,n)
x3 = np.linspace(0,1,n)

bounds_goldstein = np.array([-2,2])
x4 = np.linspace(-2,2,n)
##########################################################
training_samples = 80
testing_samples = 20
##########################################################
noise = np.random.randn(*x4.shape,1)* np.sqrt(5)
noise60 = np.random.choice(noise[0], training_samples)[:,None]
noise40 = np.random.choice(noise[0], training_samples)[:,None]
z = [1,2,3,4,5]
z = np.array(z)

#%%% Functions Sandia -Sin - true
def fsin(i):
    if i==1:
        return np.sin(2*pi*x3-pi)+7*(np.sin(2*pi*x2-pi))**2 
    elif i==2:
        return np.sin(2*pi*x3-pi)+7*(np.sin(2*pi*x2-pi))**2 + 12*(np.sin(2*pi*x3-pi)) 
    elif i==3:
        return np.sin(2*pi*x3-pi)+7*(np.sin(2*pi*x2-pi))**2 + 0.5*(np.sin(2*pi*x3-pi)) 
    elif i==4:
        return np.sin(2*pi*x3-pi)+7*(np.sin(2*pi*x2-pi))**2 + 8*(np.sin(2*pi*x3-pi))
    elif i==5:
        return np.sin(2*pi*x3-pi)+7*(np.sin(2*pi*x2-pi))**2 + 3.5*(np.sin(2*pi*x3-pi)) 
    else:
        pass
#%% Sandia - sin- train
def fsin_train(i):
    if i==1:
        return np.sin(2*pi*x3_train-pi)+7*(np.sin(2*pi*x2_train-pi))**2 
    elif i==2:
        return np.sin(2*pi*x3_train-pi)+7*(np.sin(2*pi*x2_train-pi))**2 + 12*(np.sin(2*pi*x3_train-pi)) 
    elif i==3:
        return np.sin(2*pi*x3_train-pi)+7*(np.sin(2*pi*x2_train-pi))**2 + 0.5*(np.sin(2*pi*x3_train-pi)) 
    elif i==4:
        return np.sin(2*pi*x3_train-pi)+7*(np.sin(2*pi*x2_train-pi))**2 + 8*(np.sin(2*pi*x3_train-pi))
    elif i==5:
        return np.sin(2*pi*x3_train-pi)+7*(np.sin(2*pi*x2_train-pi))**2 + 3.5*(np.sin(2*pi*x3_train-pi)) 
    else:
        pass
#%% Sandia sin - test
def fsin_test(i):
    if i==1:
        return np.sin(2*pi*x3_test-pi)+7*(np.sin(2*pi*x2_test-pi))**2 
    elif i==2:
        return np.sin(2*pi*x3_test-pi)+7*(np.sin(2*pi*x2_test-pi))**2 + 12*(np.sin(2*pi*x3_test-pi)) 
    elif i==3:
        return np.sin(2*pi*x3_test-pi)+7*(np.sin(2*pi*x2_test-pi))**2 + 0.5*(np.sin(2*pi*x3_test-pi)) 
    elif i==4:
        return np.sin(2*pi*x3_test-pi)+7*(np.sin(2*pi*x2_test-pi))**2 + 8*(np.sin(2*pi*x3_test-pi))
    elif i==5:
        return np.sin(2*pi*x3_test-pi)+7*(np.sin(2*pi*x2_test-pi))**2 + 3.5*(np.sin(2*pi*x3_test-pi)) 
    else:
        pass    
#%%  Sandia - Goldstien -true
def goldstein(i, x4): 
    if i==-2:
        return (1+(i+x4+1)**2 * (19-14*i+3*i**2-14*x4 + 6*i*x4 + 3*x4**2))* (30+(2*i-3*x4)**2 * (18-32*i+12*i**2 + 48*x4 - 36*i*x4 + 27*x4**2))
    elif i == 0:
        return (1+(i+x4+1)**2 * (19-14*i+3*i**2-14*x4 + 6*i*x4 + 3*x4**2))* (30+(2*i-3*x4)**2 * (18-32*i+12*i**2 + 48*x4 - 36*i*x4 + 27*x4**2))
    elif i==2:
        return (1+(i+x4+1)**2 * (19-14*i+3*i**2-14*x4 + 6*i*x4 + 3*x4**2))* (30+(2*i-3*x4)**2 * (18-32*i+12*i**2 + 48*x4 - 36*i*x4 + 27*x4**2))
#%% Sandia goldstein -train
def goldstein_train(i, x4_train): 
    if i==-2:
        return (1+(i+x4_train+1)**2 * (19-14*i+3*i**2-14*x4_train + 6*i*x4_train + 3*x4_train**2))* (30+(2*i-3*x4_train)**2 * (18-32*i+12*i**2 + 48*x4_train - 36*i*x4_train + 27*x4_train**2))
    elif i == 0:
        return (1+(i+x4_train+1)**2 * (19-14*i+3*i**2-14*x4_train + 6*i*x4_train + 3*x4_train**2))* (30+(2*i-3*x4_train)**2 * (18-32*i+12*i**2 + 48*x4_train - 36*i*x4_train + 27*x4_train**2))
    elif i==2:
        return (1+(i+x4_train+1)**2 * (19-14*i+3*i**2-14*x4_train + 6*i*x4_train + 3*x4_train**2))* (30+(2*i-3*x4_train)**2 * (18-32*i+12*i**2 + 48*x4_train - 36*i*x4_train + 27*x4_train**2))    
    
#%% Goldstein lambda
f_output1 = lambda x :(1+(-2+x+1)**2*(19-14*-2+3*-2**2-14*x + 6*-2*x + 3*x**2))* (30+(2*-2-3*x)**2 * (18-32*-2+12*-2**2 + 48*x - 36*-2*x + 27*x**2)) 
f_output2 = lambda x :(1+(0+x+1)**2 * (19-14*0+3*0**2-14*x + 6*0*x + 3*x**2))* (30+(2*0-3*x)**2 * (18-32*0+12*0**2 + 48*x - 36*0*x + 27*x**2))
f_output3 = lambda x :(1+(2+x+1)**2 * (19-14*2+3*2**2-14*x + 6*2*x + 3*x**2))* (30+(2*2-3*x)**2 * (18-32*2+12*2**2 + 48*x - 36*2*x + 27*x**2))

#f_output1 = lambda x : 4 * 0.0023*0.5*0.988*x**2
#f_output2 = lambda x :15 * 0.0023*0.5*0.988*x**2
#f_output3 = lambda x :30 * 0.0023*0.5*0.988*x**2


#%% Zhou 2012
def fcos(j,x2):
    if j==1:
        return np.cos(6.8*pi*x2/2)
    if j==2:
        return -np.cos(7*pi*x2/2)
    if j==3:
        return np.cos(7.2*pi*x2/2)
#%% Plotting

"""
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

for i in range(1,6):
    plt.plot(x2,fsin(i),label ="x1="+str(i))
    plt.scatter(x2,fsin(i)+gaussian_noise, label = "w_N(0,0.01) at "+str(i))
    plt.margins(x=0)
    plt.legend(loc = "best")

xlim = (0,1); ylim = (-10,20)
ax.set_xlim(xlim)
ax.set_ylim(ylim)


x1 = [-2,0,2]
fig2 = plt.figure(figsize = ( 10,8 ))
ax2 = fig2.add_subplot(111)
for i in x1:
    plt.plot(x4, goldstein(i,x4), label = "x1="+str(i))
    ax2.set_ylim(0,5e5)
    ax2.set_xlim(-2,2)
    plt.margins(x=0)
    plt.legend(loc="best")
"""
"""
x5 = [1,2,3]
fig3 = plt.figure(figsize = ( 10,8 ))
ax3 = fig3.add_subplot(111)
for i in x5:
    plt.plot(x2, fcos(i,x2), label = "z1="+str(i))
    ax3.set_ylim(-1.5,1.5)
    ax3.set_xlim(0,1)
    plt.margins(x=0)
    plt.legend(loc="best")
"""

#%% plot train and test


def plot_2outputs(m,xlim,ylim):
    fig1 = plt.figure(figsize=(12,10))
    #Output 1
    ax2 = fig1.add_subplot(311)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 1')
    #slices = GPy.util.multioutput.get_slices([x4_train1, x4_train2, x4_train3])
    m.plot(plot_limits=xlim,fixed_inputs=[(1,0)],which_data_rows=slice(0,training_samples),ax=ax2,label='icm')
    #plt.plot(np.linspace(-2,2,180), ys[60], label="test1")
    ax2.plot(x4_test1[:,:1],f_output1(x4_test1[:,:1]),'rx',mew=1.5, label='test x=-2')
    plt.legend(loc='best')
    plt.tight_layout()
    #Output 2
    ax3 = fig1.add_subplot(312)
    ax3.set_xlim(xlim)
    ax3.set_title('Output 2')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,1)],which_data_rows=slice(training_samples,2*training_samples),ax=ax3, label='icm')
    #plt.plot(np.linspace(-2,2,180), ys[60:120], label="test2")
    ax3.plot(x4_test2[:,:1],f_output2(x4_test2[:,:1]),'rx',mew=1.5, label='test x=0')
    plt.legend(loc='best')
    plt.tight_layout()
    #Output 3
    ax4 = fig1.add_subplot(313)
    ax4.set_xlim(xlim)
    ax4.set_title('Output 3')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,2)],which_data_rows=slice(2*training_samples,3*training_samples),ax=ax4,label='icm')
    #plt.plot(np.linspace(-2,2,180), ys[120:180], label="test3")
    ax4.plot(x4_test3[:,:1],f_output3(x4_test3[:,:1]),'rx',mew=1.5, label='test x=2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

#%% Building Co-regionalized model - fsin_
"""
K=GPy.kern.RBF(1)
B = GPy.kern.Coregionalize(input_dim=2,output_dim=5) 
multkernel = K.prod(B,name='B.K')
print(multkernel)

print( 'W matrix\nm',B.W)
print( '\nkappa vector\n',B.kappa)
print( '\nB matrix\n',B.B)

icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=5,kernel=K)
print('icm \n',icm)

X1 = [x2_train, x3_train, x2_train, x3_train, x2_train]
X1 = np.array(X1)
X1 = X1.reshape(np.size(X1),1)
Y1 = [fsin_train(1), fsin_train(2), fsin_train(3), fsin_train(4), fsin_train(5)]
Y1 = np.array(Y1)
Y1 = Y1.reshape(np.size(Y1),1)
#Y1 = list(Y1)

m = GPy.models.GPCoregionalizedRegression(X1,Y1,kernel=icm)
m['.*rbf.var'].constrain_fixed(1.)
m.optimize()
print("\n")
print(m)
plot_2outputs(m,xlim=(0,1),ylim=(-10,12))
plt.show()
"""

#%%Extras
#a,b,c = GPy.plotting.gpy_plot.gp_plots.get_x_y_var(m) #gives X,X_variance, Y

#%% True fucntion plot


X3 = pyd.lhs(3, training_samples, criterion='centermaximin') # 3 cols of 60 samples
X4= X3*(np.max(bounds)-np.min(bounds))+np.min(bounds)

x4_train1 = X4[:,0]
x4_train1 = np.reshape(x4_train1, (len(x4_train1),1))
x4_train2 = X4[:,1]
x4_train2 = np.reshape(x4_train2, (len(x4_train2),1))
x4_train3 = X4[:,2]
x4_train3 = np.reshape(x4_train3, (len(x4_train3),1))

Xb = pyd.lhs(3,testing_samples)
Xc = Xb*(np.max(bounds)-np.min(bounds))+np.min(bounds)

x4_test1 = Xc[:,0]
x4_test1 = np.reshape(x4_test1, (len(x4_test1),1))
x4_test2 = Xc[:,1]
x4_test2 = np.reshape(x4_test2, (len(x4_test2),1))
x4_test3 = Xc[:,2]
x4_test3 = np.reshape(x4_test3, (len(x4_test3),1))


fig2 = pb.figure(figsize=(12,8))
ax1 = fig2.add_subplot(111)
ax1.plot(x4_train1[:,:1],f_output1(x4_train1[:,:1]),'kx',mew=1.5,label='Train set x=-2')
ax1.plot(x4_train2[:,:1],f_output2(x4_train2[:,:1]),'kx',mew=1.5,label='Train set x=0')
ax1.plot(x4_train3[:,:1],f_output3(x4_train3[:,:1]),'kx',mew=1.5,label='Train set x=2')
ax1.plot(x4,f_output1(x4),mew=1.5,label='True fx x=-2')
ax1.plot(x4,f_output2(x4),mew=1.5,label='True fx x=0')
ax1.plot(x4,f_output3(x4),mew=1.5,label='True fx x=2')
ax1.plot(x4_test1[:,:1],f_output1(x4_test1[:,:1]),'o',mew=3,label='test set x=-2')
ax1.plot(x4_test2[:,:1],f_output2(x4_test2[:,:1]),'o',mew=3,label='test set x=0')
ax1.plot(x4_test3[:,:1],f_output3(x4_test3[:,:1]),'o',mew=3,label='test set x=2')
plt.legend(loc='best')
plt.show()


y_true = [f_output1(x4), f_output2(x4), f_output3(x4)]
K=GPy.kern.RBF(1)
#K = GPy.kern.Matern32(1)
#%%% MOGP
B = GPy.kern.Coregionalize(input_dim=1,output_dim=3) 
multkernel = K.prod(B,name='B.K')
print(multkernel)

print( 'W matrix\nm',B.W)
print( '\nkappa vector\n',B.kappa)
print( '\nB matrix\n',B.B)

icm = GPy.util.multioutput.ICM(input_dim=1,num_outputs=3,kernel=K)
print('icm \n',icm)

a = [x4_train1, x4_train2, x4_train3]
b = [f_output1(x4_train1), f_output2(x4_train2), f_output3(x4_train3)]

m = GPy.models.GPCoregionalizedRegression(a,b,kernel=icm) #Training set!

#m['.*Mat32.var'].constrain_fixed(1.)
m['.*rbf.var'].constrain_fixed(1.)
m.optimize()
print(m)


#%% Prediction 

Xb1 = pyd.lhs(3,testing_samples)   #should be unseen data 
Xc1 = Xb1*(np.max(bounds)-np.min(bounds))+np.min(bounds)
Xc1 = Xb1*(np.max(bounds1)-np.min(bounds1))+np.min(bounds1)


t21 = Xc1[:,0]
t21 = np.reshape(t21, (len(t21),1))
t22 = Xc1[:,1]
t22 = np.reshape(t22, (len(t22),1))
t23 = Xc1[:,2]
t23 = np.reshape(t23, (len(t23),1))

#---------------------------------------
a1 = np.r_[t21,t22,t23]
b1 = np.zeros((3*testing_samples, 1), dtype=int)
xnew = np.append(a1,b1,axis=1)
#---------------------------------------
ytest = np.r_[f_output1(t21), f_output2(t22), f_output3(t23)]
ytest1 = np.zeros((3*testing_samples,1), dtype=int)
ynew = np.append(ytest, ytest1, axis=1)
#------------------------------------

ymeta = {'output_index': xnew[:,-1].astype(int)}
ys, var = m.predict(xnew, Y_metadata=ymeta) # y1 y2 y3 = MOGP

#%% predicting f1

order1 = np.argsort(xnew[0:testing_samples,0])
xs1 = np.array(xnew[0:testing_samples,0])[order1]
ys1 = np.array(ys[0:testing_samples])[order1] 

order2 = np.argsort(xnew[testing_samples:2*testing_samples,0])
xs2 = np.array(xnew[testing_samples:2*testing_samples,0])[order2]
ys2 = np.array(ys[testing_samples:2*testing_samples])[order2] #mogp sliced outputs

order3 = np.argsort(xnew[2*testing_samples:3*testing_samples,0])
xs3 = np.array(xnew[2*testing_samples:3*testing_samples,0])[order3]
ys3 = np.array(ys[2*testing_samples:3*testing_samples])[order3]

r21 = r2_score(f_output1(xnew[0:testing_samples,0]),ys[0:testing_samples]) #between output values and mean of model ??
r22 = r2_score(f_output2(xnew[testing_samples:2*testing_samples,0]),ys[testing_samples:2*testing_samples])
r23 = r2_score(f_output3(xnew[2*testing_samples:3*testing_samples,0]),ys[2*testing_samples:3*testing_samples])
"""
#%%
## all predictions
fig6 = plt.figure(figsize=(12,10))
ax6 = fig6.add_subplot(311)
ax6.set_title('Testing set with Coregionalized model')
r21 = r2_score(f_output1(xnew[0:testing_samples,0]),ys[0:testing_samples])

ax6.plot(xs1, ynew[0:testing_samples,0], 'kx--', label='f1(test)') #MOGP x, MOGP testing y
ax6.plot(xs1, ys1,'o-', label='m') # x, MOGP prediction with 
plt.legend(loc='best')
plt.tight_layout()
#-----------------------
ax7 = fig6.add_subplot(312)
ax7.plot(xs2, ynew[testing_samples:2*testing_samples, 0], 'kx--', label='f2(test)');
ax7.plot(xs2, ys2,'o-', label='m')
r22 = r2_score(f_output2(xs2),ys[testing_samples:2*testing_samples])
plt.legend(loc='best')
plt.tight_layout()
#-------------------------
ax8 = fig6.add_subplot(313)
ax8.plot(xs3, ynew[2*testing_samples:3*testing_samples, 0], 'kx--', label='f3(test)');
ax8.plot(xs3, ys3,'o-', label='m')
r23 = r2_score(f_output3(xnew[2*testing_samples:3*testing_samples,0]),ys[2*testing_samples:3*testing_samples])
plt.legend(loc='best')
plt.tight_layout()
"""

#%% 

fig9=plt.figure(figsize=(5,5), tight_layout=True)
plt.plot(np.linspace(min(ys1), max(ys1), 100), np.linspace(min(f_output1(xs1)[order1]), max(f_output1(xs1)[order1]), 100)) #straight line
plt.scatter(ys1,ynew[0:testing_samples,0] , label='R2-MO1 {}'.format(str(r21)));plt.legend(loc='best');plt.show()
#plt.text(-4e5,2e5,'NRMSE={}'.format(str(N1)))
#rsme1 = np.sqrt(mean_squared_error(f_output1(xnew[0:testing_samples,0]), ys[0:testing_samples])/(max(ys[0:testing_samples]) - min(ys[0:testing_samples])))


fig10=plt.figure(figsize=(5,5), tight_layout=True)
plt.plot(np.linspace(min(ys2), max(ys2), 100), np.linspace(min(f_output2(xs2)[order2]), max(f_output2(xs2)[order2]), 100))
plt.scatter(ys2,ynew[testing_samples:2*testing_samples,0], label='R2-MO2 {}'.format(str(r22)));plt.legend(loc='best');plt.show()
#rsme2 = np.sqrt(mean_squared_error(f_output2(xnew[testing_samples:2*testing_samples,0]), ys[testing_samples:2*testing_samples])/ (max(ys[testing_samples:2*testing_samples]) - min(ys[testing_samples:2*testing_samples])))
#plt.text(-12e4,12e4,'NRMSE ={}'.format(str(N2)))

fig11=plt.figure(figsize=(5,5), tight_layout=True)
plt.plot(np.linspace(min(ys3), max(ys3), 100), np.linspace(min(f_output3(xs3)[order3]), max(f_output3(xs3)[order3]), 100))
plt.scatter(ys3, ynew[2*testing_samples:3*testing_samples,0], label='R2-MO3 {}'.format(str(r23))) ;plt.legend(loc='best'); plt.show()
#rsme3 = np.sqrt(mean_squared_error(f_output3(xnew[2*testing_samples:3*testing_samples,0]), ys[2*testing_samples:3*testing_samples])/ (max(ys[2*testing_samples:3*testing_samples]) - min(ys[2*testing_samples:3*testing_samples])))
#plt.text(-10e4,2.5e5,'NRMSE ={}'.format(str(N3)))

"""
    print("RMSE is: ", np.sqrt(mean_squared_error(f_output1(xnew[0:testing_samples,0]), ys[0:testing_samples])))
    print("NRMSE is: ", np.sqrt(mean_squared_error(f_output1(xnew[0:testing_samples,0]), ys[0:testing_samples]))/ (max(f_output1(xnew[0:testing_samples,0])) - min(f_output1(xnew[0:testing_samples,0]))))
    print("MSE is: ", mean_squared_error(f_output1(xnew[0:testing_samples,0]),ys[0:testing_samples]))
    print("MAE  is: ", mean_absolute_error(f_output1(xnew[0:testing_samples,0]),ys[0:testing_samples]))   
    print("R21 is: ", r2_score(f_output1(xnew[0:testing_samples,0]),ys[0:testing_samples]))
    
    print("R22 is: ", r2_score(f_output2(xs2),ys[testing_samples:2*testing_samples]))
    print("R23 is: ", r2_score(f_output3(xnew[2*testing_samples:3*testing_samples,0]),ys[2*testing_samples:3*testing_samples])
          
          
    mean, _ = m.predict_y(xnew)
"""

#print(m.Y.T)
#rm = rmse(m.Y[0:60,:], f_output1(x4_train1))

plot_2outputs(m, xlim=xlim,ylim=ylim)
plt.legend(loc='best')
plt.show()

#%% Compare with SOGP

k1 = GPy.kern.RBF(1)
#k1 = GPy.kern.Matern32(1)
m1 = GPy.models.GPRegression(x4_train1 , f_output1(x4_train1) ,kernel=k1.copy()) #g1
m1.optimize('bfgs')

#%%
yso1 = m1.predict(xnew[0:testing_samples],  Y_metadata=ymeta)

#%%
m2 = GPy.models.GPRegression(x4_train2,f_output2(x4_train2),kernel=k1.copy()) #g2
m2.optimize('bfgs')
yso2 = m2.predict(xnew[testing_samples:2*testing_samples],  Y_metadata=ymeta)

#%%

m3 = GPy.models.GPRegression(x4_train3,f_output3(x4_train3),kernel=k1.copy()) #g3
m3.optimize('bfgs')
yso3 = m3.predict(xnew[2*testing_samples:3*testing_samples], Y_metadata=ymeta)

#%% Argsort SOGP stuff

orderso1 = np.argsort(xnew[0:testing_samples,0])
orderso2 = np.argsort(xnew[testing_samples:2*testing_samples,0])
orderso3 = np.argsort(xnew[2*testing_samples:3*testing_samples,0]) 
yso1 = np.array(yso1[0])[orderso1]
yso2 = np.array(yso2[0])[orderso2]
yso3 = np.array(yso3[0])[orderso3]

#%%
#%%% PLOTTING!!!!!!

fig6 = plt.figure(figsize=(12,10))
ax6 = fig6.add_subplot(311)
ax6.set_title('Individual model predictions on testing set with Coregionalized model')
ax6.plot(xs1, yso1, 'kx--', label='m1-SO') #train x, SOGP mean
ax6.plot(xs1, ys1,'o-', label='m-MO') # x, MOGP mean
plt.legend(loc='lower right')
plt.tight_layout()
#-----------------------
ax7 = fig6.add_subplot(312)
ax7.plot(xs2,yso2, 'kx--', label='m2-SO');
ax7.plot(xs2, ys2,'o-', label='m-MO')
plt.legend(loc='lower right')
plt.tight_layout()
#-------------------------
ax8 = fig6.add_subplot(313)
ax8.plot(xs3, yso3, 'kx--', label='m3-SO');
ax8.plot(xs3, ys3,'o-', label='m-MO')
plt.legend(loc='lower right')
plt.tight_layout()


#%% Full model RMSE error and MAE and R2
#from tabulate import tabulate
N1 =  np.sqrt(mean_squared_error(ys[0:testing_samples],ynew[0:testing_samples,0]))#/ (max(ynew[0:testing_samples,0])- min(ys[0:testing_samples,0])) #MOGP
N11 =  np.sqrt(mean_squared_error(yso1, ynew[0:testing_samples,0]))#/ (max(ynew[0:testing_samples,0])- min(ynew[0:testing_samples,0])) #SOGP ##RMSE is between mean and fvalue
MAE1 = mean_absolute_error(ys[0:testing_samples], ynew[0:testing_samples,0])
MAE11 = mean_absolute_error(yso1,ynew[0:testing_samples,0])
r21 = r2_score(f_output1(xnew[0:testing_samples,0]),ys[0:testing_samples]) #between output values and mean of model ??
rso21 = r2_score(f_output1(xnew[0:testing_samples,0]),yso1)

N2 =  np.sqrt(mean_squared_error(ys[testing_samples:2*testing_samples],ynew[testing_samples:2*testing_samples,0]))#/ (max(ynew[testing_samples:2*testing_samples,0])- min(ynew[testing_samples:2*testing_samples,0]))
N22 = np.sqrt(mean_squared_error(yso2,ynew[testing_samples:2*testing_samples,0]))#/ (max(ynew[testing_samples:2*testing_samples,0])- min(ynew[testing_samples:2*testing_samples,0]))
MAE2 = mean_absolute_error(ys[testing_samples:2*testing_samples],ynew[testing_samples:2*testing_samples,0])
MAE22 = mean_absolute_error(yso2,ynew[testing_samples:2*testing_samples,0])
r22 = r2_score(f_output2(xnew[testing_samples:2*testing_samples,0]),ys[testing_samples:2*testing_samples])
rso22 = r2_score(f_output2(xnew[testing_samples:2*testing_samples,0]),yso2)

N3 =  np.sqrt(mean_squared_error(ys[2*testing_samples:3*testing_samples],ynew[2*testing_samples:3*testing_samples,0]))#/ (max(ynew[2*testing_samples:3*testing_samples,0])- min(ynew[2*testing_samples:3*testing_samples,0]))
N33 = np.sqrt(mean_squared_error(yso3,ynew[2*testing_samples:3*testing_samples,0]))#/ (max(ynew[2*testing_samples:3*testing_samples,0])- min(ynew[2*testing_samples:3*testing_samples,0]))
MAE3 = mean_absolute_error(ys[2*testing_samples:3*testing_samples], ynew[2*testing_samples:3*testing_samples,0])
MAE33 = mean_absolute_error(yso3, ynew[2*testing_samples:3*testing_samples,0])
r23 = r2_score(f_output3(xnew[2*testing_samples:3*testing_samples,0]),ys[2*testing_samples:3*testing_samples])
rso23 = r2_score(f_output3(xnew[2*testing_samples:3*testing_samples,0]),yso3)
#%%%

fig12 = pb.figure(figsize=(12,8))
#Output 1
ax1 = fig12.add_subplot(311)
m1.plot(plot_limits=xlim,ax=ax1)
ax1.plot(x4_test1[:,:1],f_output1(x4_test1[:,:1]),'rx',mew=1.5)
ax1.set_title('Output 1-SOGP')
#Output 2
ax2 = fig12.add_subplot(312)
m2.plot(plot_limits=xlim,ax=ax2)
ax2.plot(x4_test2[:,:1],f_output2(x4_test2[:,:1]),'rx',mew=1.5)
ax2.set_title('Output 2-SOGP')
#output3
ax3 = fig12.add_subplot(313)
m3.plot(plot_limits=xlim,ax=ax3)
ax3.plot(x4_test3[:,:1], f_output3(x4_test3[:,:1]),'rx',mew=1.5)
ax3.set_title('Output 3-SOGP')
plt.legend(loc='best')
plt.tight_layout()

#%%% SOGP- R2 assessment


fig13=plt.figure(figsize=(5,5), tight_layout=True)
plt.plot(np.linspace(min(yso1), max(yso1), 100), np.linspace(min(f_output1(xs1)[order1]), max(f_output1(xs1)[order1]), 100)) #straight line
plt.scatter(yso1,ynew[0:testing_samples,0] , label='R2-SO1 {}'.format(str(rso21)));plt.legend(loc='best');plt.show()
#plt.text(-4e5,2e5,'NRMSE={}'.format(str(N1)))
#rsme1 = np.sqrt(mean_squared_error(f_output1(xnew[0:testing_samples,0]), ys[0:testing_samples])/(max(ys[0:testing_samples]) - min(ys[0:testing_samples])))


fig14=plt.figure(figsize=(5,5), tight_layout=True)
plt.plot(np.linspace(min(yso2), max(yso2), 100), np.linspace(min(f_output2(xs2)[order2]), max(f_output2(xs2)[order2]), 100))
plt.scatter(yso2,ynew[testing_samples:2*testing_samples,0], label='R2-SO2 {}'.format(str(rso22)));plt.legend(loc='best');plt.show()
#rsme2 = np.sqrt(mean_squared_error(f_output2(xnew[testing_samples:2*testing_samples,0]), ys[testing_samples:2*testing_samples])/ (max(ys[testing_samples:2*testing_samples]) - min(ys[testing_samples:2*testing_samples])))
#plt.text(-12e4,12e4,'NRMSE ={}'.format(str(N2)))

fig15=plt.figure(figsize=(5,5), tight_layout=True)
plt.plot(np.linspace(min(yso3), max(yso3), 100), np.linspace(min(f_output3(xs3)[order3]), max(f_output3(xs3)[order3]), 100))
plt.scatter(yso3, ynew[2*testing_samples:3*testing_samples,0], label='R2-SO3 {}'.format(str(rso23))) ;plt.legend(loc='best'); plt.show()


#%%
"""
fig13=plt.figure()
ax5= fig13.add_subplot(111)
ax5 = plt.scatter(x4_train1, m.Y[0:training_samples,:], s=10, c='b', marker='s', label='m')
ax5 = plt.scatter(x4, f_output1(x4),s=10, c='r', marker='o', label='f(x4)')
plt.show()
plt.legend(loc='best')
"""




"""
X1 = np.array(X1)
X1 = np.reshape(X1, (len(X1),1))
X2 = x4_train
X2 = np.reshape(X2, (len(X2),1))

yg1 = goldstein(X1[0],x4_train)
yg1 = np.reshape(yg1, (len(yg1),1))
yg2 = goldstein(X1[1],x4_train)
yg2 = np.reshape(yg2, (len(yg2),1))
yg3 = goldstein(X1[2],x4_train)
yg3 = np.reshape(yg3, (len(yg3),1))
Xg = [X1,X2]
Yg = [yg1, yg2, yg3]


X1widx = np.c_[x4_train, np.ones(x4_train.shape[0])*0]
X2widx = np.c_[x4_train, np.ones(x4_train.shape[0])*1]
X3widx = np.c_[x4_train, np.ones(x4_train.shape[0])*2]
X1 = np.r_[X1widx, X2widx, X3widx]

yg1 = np.array([], dtype=float)
yg1 = goldstein(-2, x4_train)
yg1 = [yg1]

yg2 = np.array([], dtype=float)
yg2 = goldstein(0, x4_train)
yg2 = [yg2]

yg3 = np.array([], dtype=float)
yg3 = goldstein(2, x4_train)
yg3 = [yg3]


Y1 = np.c_[yg1, yg2, yg3]
Y1=Y1.T
#Y1 = np.array(Y1)
#Y1 = Y1.reshape(np.size(Y1),1)
#Y1 = list(Y1)


#%% w = [x.T, z.T].T = all quantitatve(cont) and qualitative(disc) factors
w = np.array([],dtype=float)
w = [x2.T , x3.T, z.T]
#w2 = np.array(w1)
#w = w2.T
#print(w)
#print(X)
"""

#%% plotting example self -sufficient
"""
np.random.seed(111)
import matplotlib.pyplot as plt
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
#matplotlib.rcParams[u'figure.figsize'] = (4,3)
matplotlib.rcParams[u'text.usetex'] = False
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X = np.random.uniform(-2, 2, (40, 1))
    f = .2 * np.sin(1.3*X) + 1.3*np.cos(2*X)
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.SparseGPRegression(X, Y, X_variance=np.ones_like(X)*[0.06])
    #m.optimize()
    m.plot_data(label='plot_data')
    plt.legend(loc='best')
    m.plot_mean(label='plot_mean')
    plt.legend(loc='best')
    m.plot_confidence(label='confidence')
    plt.legend(loc='best')
    m.plot_density(label='density')
    plt.legend(loc='best')
    m.plot_errorbars_trainset(label='errorbars_trainset')
    plt.legend(loc='best')
    m.plot_samples(label='samples')
    plt.legend(loc='best')
    m.plot_data_error(label='data_error')
    plt.legend(loc='best')
    m.plot_inducing(marker='s')
    plt.tight_layout()
    plt.show()
"""


#%%% Plottting 3D self contained
"""
np.random.seed(11111)
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams[u'text.usetex'] = False
X = np.random.uniform(-2, 2, (40, 2))
f = .2 * np.sin(1.3*X[:,[0]]) + 1.3*np.cos(2*X[:,[1]])
Y = f+np.random.normal(0, .1, f.shape)
m = GPy.models.SparseGPRegression(X, Y)
m.likelihood.variance = .1
m.plot_samples(projection='3d', samples=1, label='samples')
m.plot_samples(projection='3d', plot_raw=False, samples=1)
plt.close('all')
m.plot_data(projection='3d')
m.plot_mean(projection='3d', rstride=10, cstride=10, label='mean')
m.plot_inducing(projection='3d', label='plot_inducing')
plt.show()

"""

"""
#self- sparesed example
def coregionalization_sparse(optimize=True, plot=True):
    #build a design matrix with a column of integers indicating the output
    X1 = np.random.rand(50, 1) * 8
    X2 = np.random.rand(30, 1) * 5

    #build a suitable set of observed variables
    Y1 = np.sin(X1) + np.random.randn(*X1.shape) * 0.05
    Y2 = np.sin(X2) + np.random.randn(*X2.shape) * 0.05 + 2.

    m = GPy.models.SparseGPCoregionalizedRegression(X_list=[X1,X2], Y_list=[Y1,Y2])

    if optimize:
        m.optimize('bfgs', max_iters=100)

    if plot:
        slices = GPy.util.multioutput.get_slices([X1,X2])
        m.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0})
        m.plot(fixed_inputs=[(1,1)],which_data_rows=slices[1],Y_metadata={'output_index':1},ax=pb.gca())
        pb.ylim(-3,)

    plt.show()
    return m
    
"""
