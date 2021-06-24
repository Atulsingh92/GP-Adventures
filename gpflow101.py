import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gpflow.ci_utils import ci_niter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gpflow.utilities import print_summary
import math 
np.set_printoptions(precision=4)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams['legend.fontsize']  = 14
plt.rcParams.update({'font.size': 14})
np.random.seed(12)


#%% ############################## MOGP 

# make a dataset with two outputs, correlated, heavy-tail noise. One has more noise than the other.
sampling_bounds = [1,7]
x1 = np.random.rand(3,1) # Observed locations for first outputx
x2 = np.random.rand(5,1)  # 0.5  # Observed locations for second output

#x1 = [[0.1912], [0.31214], [0.243123], [0.4234], [0.73452], [0.942234]]
#x2 = [[0.0424], [0.124457], [0.35242], [0.69875], [0.856478], [0.88535]]

X1 = x1*(np.max(sampling_bounds)-np.min(sampling_bounds))+np.min(sampling_bounds)
X2 = x2*(np.max(sampling_bounds)-np.min(sampling_bounds))+np.min(sampling_bounds)
#addition = np.linspace(4,6,20)
#b = addition.reshape((20,1))

#X2 = np.vstack((X2, b))
#Y1 = np.sin(6*X1)# + np.random.randn(*X1.shape) * 0.03
#Y2 = np.sin(6*X2 + 0.7)# + np.random.randn(*X2.shape) * 0.1

Y1 = lambda X1: np.sin(X1)# + np.random.randn(*X1.shape) * np.sqrt(0.01)
Y2 = lambda X2: np.sin(X2) + 0.2*np.cos(3*X2)# + np.random.randn(*X2.shape) *np.sqrt(0.01)

#Y1 = lambda X1: np.sin(2*math.pi*X1)
#Y2 = lambda X2: (X2/4 - np.sqrt(2))*np.sin(2*math.pi*X2+3*math.pi)**3

#order1 = np.argsort(X1.flatten())
#xsorted1 = X1.flatten()[order1]
#ysorted1 = Y1.flatten()[order1]

#order2 = np.argsort(X2.flatten())
#xsorted2 = X2.flatten()[order2]
#ysorted2 = Y2.flatten()[order2]

plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0,7,100), Y1(np.linspace(0,7,100)), color='red', label='y1-true')
plt.plot(X1,Y1(X1),'ro' , label='y1 samples', ms=7, mew =1.5, markeredgecolor='black')
plt.plot(np.linspace(0,7,100), Y2(np.linspace(0,7,100)), color='blue',label='y2-true')
plt.plot(X2,Y2(X2),'bs', label='y2 samples', ms=7, mew=1.5, markeredgecolor='black')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('True function and sample space')
plt.legend(loc='best')
plt.margins(x=0)
plt.tight_layout()
plt.show()


# Augment the input with ones or zeros to indicate the required output dimension
X_augmented = np.vstack((np.hstack((X1, np.zeros_like(X1))), np.hstack((X2, np.ones_like(X2)))))

# Augment the Y data with ones or zeros that specify a likelihood from the list of likelihoods
Y_augmented = np.vstack((np.hstack((Y1(X1), np.zeros_like(Y1(X1)))), np.hstack((Y2(X2), np.ones_like(Y2(X2))))))

output_dim = 2  # Number of outputs
rank = 1  # Rank of W

# Base kernel
#k = gpflow.kernels.Matern32(active_dims=[0])
#k = gpflow.kernels.SquaredExponential()
k = gpflow.kernels.RBF()
# Coregion kernel
coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

kern = k * coreg

# This likelihood switches between Gaussian noise with different variances for each f_i:
lik = gpflow.likelihoods.SwitchedLikelihood(
    [gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()])

# now build the GP model as normal
m = gpflow.models.VGP((X_augmented, Y_augmented), kernel=kern, likelihood=lik)

# fit the covariance function parameters
maxiter = ci_niter(10000)
gpflow.optimizers.Scipy().minimize(
    m.training_loss, m.trainable_variables, options=dict(maxiter=maxiter),
    method="L-BFGS-B")

def plot_gp(x, mu, var, color, label):
    plt.plot(x, mu, ls='dashed', color=color, lw=2, label=label)
    plt.fill_between(
        x[:, 0],
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color=color,
        alpha=0.2,
        label='Uncertainty-MO',
    )

def plot(m):
    plt.figure(figsize=(12, 6))
    Xtest = np.linspace(0, 7, 50)[:, None]
    (line,) = plt.plot(X1, Y1(X1), "ro", ms=7, mew=1, color='purple' , markeredgecolor='black')
    mu, var = m.predict_f(np.hstack((Xtest, np.zeros_like(Xtest))))
    plot_gp(Xtest, mu, var, line.get_color(), "Y1")

    (line,) = plt.plot(X2, Y2(X2), "bs", ms=7, mew=1, color='orange', markeredgecolor='black')
    mu, var = m.predict_f(np.hstack((Xtest, np.ones_like(Xtest))))
    plot_gp(Xtest, mu, var, line.get_color(), "Y2")
    plt.margins(x=0)
    plt.legend()
    plt.xlabel('x_test')
    plt.ylabel('mean prediction on x_test)')
    plt.title('MOGP model prediction on test set  for two functions')
    plt.tight_layout()
    plt.show()


plot(m)

Xtest = np.linspace(0, 7, 50)[:, None]
mu, var = m.predict_f(np.hstack((Xtest, np.zeros_like(Xtest))))

#########################################################################
#%%SOGP
kso = gpflow.kernels.SquaredExponential()
m1so = gpflow.models.GPR(data=(X1, Y1(X1)), kernel=kso, mean_function=None)

m2so = gpflow.models.GPR(data=(X2, Y2(X2)), kernel=kso, mean_function=None)
opt = gpflow.optimizers.Scipy()
opt_logs1 = opt.minimize(m1so.training_loss, m1so.trainable_variables, options=dict(maxiter=100))
opt_logs2 = opt.minimize(m2so.training_loss, m2so.trainable_variables, options=dict(maxiter=100))
print_summary(m1so)
print_summary(m2so)

mu1so, var1so = m1so.predict_f(Xtest)
mu2so, var2so = m2so.predict_f(Xtest)
tf.random.set_seed(1)
samples1 = m1so.predict_f_samples(Xtest, 10) #generate 10 samples from posterior
samples2 = m2so.predict_f_samples(Xtest, 10)


plt.figure(figsize=(12,6))
(line,) = plt.plot(X1, Y1(X1), 'bo', ms=7, mew=1, label='Data points', color='purple', markeredgecolor='black')
plt.plot(Xtest, Y1(Xtest), lw=2, color='orange', label='f1-true') #was orange
plt.plot(Xtest, mu1so, c='green', ls=('dashed'), lw=2,label='m1-SO')
plt.fill_between(
    Xtest[:, 0],
    mu1so[:, 0] - 1.96 * np.sqrt(var1so[:, 0]),
    mu1so[:, 0] + 1.96 * np.sqrt(var1so[:, 0]),
    color="lightgrey",
    label ='Uncertainty-SO',
    alpha=0.4)
plot_gp(Xtest, mu, var, line.get_color(), "m-MO")
#plt.plot(Xtest, samples1[:,:,0].numpy().T, linewidth=1)
plt.legend(loc='best')
plt.margins(x=0)
plt.title('SOGP1')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
(line,) = plt.plot(X2, Y2(X2), 'rs', ms=7, mew=1, label='Data points', color='orange', markeredgecolor='black')
plt.plot(Xtest, Y2(Xtest), lw=2, color='blue',label='f2-true')
plt.plot(Xtest, mu2so, c='green', ls=('dashed') , lw=2, label='m2-SO')
plt.fill_between(
    Xtest[:, 0],
    mu2so[:, 0] - 1.96 * np.sqrt(var2so[:, 0]),
    mu2so[:, 0] + 1.96 * np.sqrt(var2so[:, 0]),
    color="lightgrey",
    label ='Uncertainty-SO',
    alpha=0.4)
plot_gp(Xtest, mu, var, line.get_color(), "m-MO")
#plt.plot(Xtest, samples2[:,:,0].numpy().T, linewidth=1)
plt.legend(loc='best')
plt.title('SOGP2')
plt.margins(x=0)
plt.tight_layout()
plt.show()

#%% Errors calculations
# mu is for MOGP
#Xtest is for MOGP
rmse1 = np.mean((mu - Y1(Xtest))**2, axis=0)**0.5
rmse2 = np.mean((mu - Y2(Xtest))**2 , axis=0)**0.5
MAE1 = np.mean(np.abs(mu - Y1(Xtest)), axis=0)
MAE2 = np.mean(np.abs(mu - Y2(Xtest)), axis=0)
r21 = r2_score(Y1(Xtest), mu)
r22 = r2_score(Y2(Xtest), mu)

rmse1so = np.mean((mu1so - Y1(Xtest))**2, axis=0)**0.5
rmse2so = np.mean((mu2so - Y2(Xtest))**2, axis=0)**0.5
MAE1so = np.mean(np.abs(mu1so - Y1(Xtest)), axis=0)
MAE2so = np.mean(np.abs(mu2so - Y2(Xtest)), axis=0)
r21so = r2_score(Y1(Xtest), mu1so)
r22so = r2_score(Y2(Xtest), mu2so)

datamo = [['Output', 'RMSE', 'MAE', 'R2'],
          ['f1', rmse1, MAE1, r21],
          ['f2', rmse2, MAE2, r22]]

dataso = [['Output', 'RMSE', 'MAE', 'R2'],
          ['f1', rmse1so, MAE1so, r21so],
          ['f2', rmse2so, MAE2so, r22so]]

np.set_printoptions(precision=4)
def drawTable(dat):
    output=""
    for item in dat[0]:
        output += '\t' + str(item) + '\t\t'    
    
    output += '\n'
    output += "================================================================="
    for item in dat[1:]:
        output += '\n'
        for k in item:
            output += '\t' + str(k) + '\t'

    return print(output)

print("=============================MOGP errors=================================")
drawTable(datamo)
print("=============================SOGP errors================================")
drawTable(dataso)

fig12 = plt.figure(figsize=(12,6))
plt.title("True function at test set vs MOGP model")
ax1 = plt.subplot(121)
plt.plot(np.linspace(min(Y1(Xtest)),max(Y1(Xtest)),50), np.linspace(min(mu), max(mu), 50), c='grey')
mu, var = m.predict_f(np.hstack((Xtest, np.zeros_like(Xtest))))
plt.plot(Y1(Xtest), mu, 'o', label='R2={}'.format(str(r21)))
plt.margins(x=0)
plt.xlabel('Y1(Xtest)')
plt.ylabel('m')
plt.legend(loc='best')
plt.tight_layout()
ax2 = plt.subplot(122)
plt.plot(np.linspace(min(Y2(Xtest)),max(Y2(Xtest)), 50), np.linspace(min(mu), max(mu), 50), c='grey')
mu, var = m.predict_f(np.hstack((Xtest, np.zeros_like(Xtest))))
plt.plot(Y2(Xtest), mu, 's', label='R2={}'.format(str(r22)))
plt.margins(x=0)
plt.xlabel('Y2(Xtest)')
plt.ylabel('m')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#%% Video
