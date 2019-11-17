
import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)

#list_points = []
#for input_image in X[l1[0][:10]]:
#    contours, hierarchy = cv2.findContours(input_image.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#    contours, boundingBoxes = sort_contours(contours)
#    points_pixels = get_boxes(boundingBoxes)
#    points_pixels = heigth-points_pixels[:,1]
#    list_points.append(points_pixels)


# ----------------------------------------------------------------------
#points_pixels[:,1] = heigth-points_pixels[:,1]
#  First the noiseless case
#X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
#Train = np.array(range(0, len(points_pixels)*4, 4))
#Train = [Train for x in range(len(list_points))]

#Train = np.array(range(0, len(points_pixels)*2, 2))
#Train = [range(70) + np.sort(rng.rand(70)) for x in range(len(list_points))]
#Train = np.concatenate(Train, axis=0).reshape(-1, 1)
#
##X = np.concatenate([X, X])
## Observations
##y = f(X).ravel()
#y = [x for x in list_points]
#y = np.concatenate(y, axis=0).reshape(-1, 1)
#noise = np.random.normal(5, 10, len(y))
#y = y + noise
#y = np.concatenate([y, y+5])

XX = np.array(range(141)).reshape(-1,1)
y = points_pixels

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, len(points_pixels - 4)*4, 1000)).T

# Instantiate a Gaussian Process model

#kernel = ConstantKernel(1.0, (1e-3, 1e3)) + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

gp = GaussianProcessRegressor(kernel=kernel,
                              n_restarts_optimizer=10)
# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(XX, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
x_test = range(70) + np.sort(rng.rand(70))
x_test = x_test.reshape(-1,1) 
y_pred, mse = gp.predict(XX+100, return_std=True)
z = 1.9600 * mse

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(XX, y, 'r.', markersize=2, label='Observations')
plt.plot(XX+100, y_pred, 'k-', label='Prediction')


plt.fill(np.concatenate([x_test, x_test[::-1]]),
         np.concatenate([y_pred - z,
                        (y_pred + z)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')


#X_train = np.array(range(len(points_pixels))).reshape(-1, 1)
#X_test = np.arange(0, len(points_pixels), 0.2).reshape(-1, 1)
#
#Y_train = points_pixels
#
#
#
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import ConstantKernel, RBF
#from gaussian_processes_util import plot_gp
#
#noise = 0.4
#kernel = ConstantKernel(1.0) * RBF(length_scale=3.0)
#kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) 
#
#gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
#
## Reuse training data from previous 1D example
#gpr.fit(X_train, Y_train)
#
## Compute posterior predictive mean and covariance
#mu_s, cov_s = gpr.predict(X_test, return_cov=True)
#
## Obtain optimized kernel parameters
#l = gpr.kernel_.k2.get_params()['length_scale']
#sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])
#
## Plot the results
#plot_gp(mu_s, cov_s, X_test, X_train=X_train, Y_train=Y_train)
#
