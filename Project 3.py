import numpy as np
import matplotlib.pyplot as plt
import math

def NNetOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_subtrain):
    V_mat = np.random.normal(0.0, pow(n_hidden_units, -0.5), (n_hidden_units, np.shape(X_mat)[1]))# initialize V.mat/w.vec to some random numbers close to zero.
    w_vec = np.random.normal(0.0, pow(1, -0.5), (1, n_hidden_units))

    # X.mat / y.vec into X.subtrain / X.validation / y.subtrain / y.validation
    indices = np.random.permutation(X_mat.shape[0])
    num = int(round(X_mat.shape[0]*is_subtrain))

    rg = indices[0:num]
    X_subtrain = X_mat[rg][:]
    y_subtrain = y_vec[rg]

    X_validation = np.delete(X_mat, rg, axis=0)
    y_validation = np.delete(y_vec, rg, axis=0)

    loss_values_mean = np.zeros(max_epochs)
    loss_values1_mean = np.zeros(max_epochs)

    for k in range(max_epochs): # a single pass through the subtrain set.
        loss_values = np.zeros(len(y_subtrain))
        loss_values1 = np.zeros(len(y_validation))

        for i in range(len(y_subtrain)):
            # Forward propagation.
            a = np.dot(V_mat, X_subtrain[i, :].T)
            h = 1 / (1 + np.exp(-a))

            yp = np.dot(w_vec, h.T)

            # Back propagation.
            # compute the gradients of V.mat/w.vec.
            grad_a = -y_subtrain[i] / (1 + np.exp(y_subtrain[i] * yp))

            # update V.mat/w.vec by taking a step (scaled by step.size) in the negative gradient direction.
            grad_w = grad_a * h
            grad_h = grad_a * w_vec
            grad_a = grad_h * h * (1 - h)
            grad_a = grad_a.reshape(-1)
            grad_a1 = np.zeros((len(grad_a), 1))
            grad_a1[:, 0] = grad_a
            X_subtrain1 = np.zeros((1, len(X_subtrain[i, :])))
            X_subtrain1[0, :] = X_subtrain[i, :]
            grad_V = np.dot(grad_a1, X_subtrain1)

            w_vec -= step_size * grad_w
            V_mat -= step_size * grad_V
            #  logistic loss on the subtrain/validation sets: log[1+exp(-yf(x))]
            #loss_values[i] = math.log((1 + np.exp(-y_subtrain[i] * yp)), math.e)
        for i in range(len(y_subtrain)):
            a = np.dot(V_mat, X_subtrain[i, :].T)
            h = 1 / (1 + np.exp(-a))
            yp = np.dot(w_vec, h.T)
            loss_values[i] = math.log((1 + np.exp(-y_subtrain[i] * yp)), math.e)

        loss_values_mean[k] = np.mean(loss_values)

        for i in range(len(y_validation)):
            a1 = np.dot(V_mat, X_validation[i, :].T)
            h = 1 / (1 + np.exp(-a1))
            yp1 = np.dot(w_vec, h.T)
            loss_values1[i] = math.log((1 + np.exp(-y_validation[i] * yp1)), math.e)

        loss_values1_mean[k] = np.mean(loss_values1)

    return [loss_values_mean, loss_values1_mean, V_mat, w_vec]



np.random.seed(0)
file = 'spam.txt'
data = np.loadtxt(file, delimiter=' ', dtype=float) #, skiprows=1)

X = data[:, :-1]
y = data[:, -1]

Xm = np.zeros(len(y))
Xs = np.zeros(len(y))
for i in range(np.shape(X)[1]):
    Xm = np.mean(X[:, i])
    Xs = np.std(X[:, i], ddof=1)
    X[:, i] = (X[:, i]-Xm)/Xs

for j in range(len(y)):
    if y[j] == 0:
        y[j] = -1

indices = np.random.permutation(X.shape[0])
num = X.shape[0] // 5

rg = indices[0:num]
X_test = X[rg][:]
y_test = y[rg]

X_train = np.delete(X, rg, axis=0)
y_train = np.delete(y, rg, axis=0)

# Define number of hidden units.
n_hidden_units = 50
step_size = 0.01
max_epochs = 300

[loss_values_mean, loss_values1_mean, V_mat, w_vec] = NNetOneSplit(X_train, y_train, max_epochs, step_size, n_hidden_units, 0.6)

a = np.array(loss_values1_mean)
b = a.tolist()
best_epochs = b.index(min(b))+1
print(best_epochs)

x = range(1, max_epochs + 1)

plt.plot(x, loss_values1_mean, label="validation", color='red')
plt.plot(x, loss_values_mean, label="train", color='blue')

[loss_values_mean, loss_values1_mean, V_mat, w_vec] = NNetOneSplit(X_train, y_train, best_epochs, step_size, n_hidden_units, 0.6)
yp = np.zeros(len(y_test))
for i in range(len(y_test)):
    a = np.dot(V_mat, X_test[i, :].T)
    h = 1 / (1 + np.exp(-a))
    yp[i] = np.dot(w_vec, h.T)
for j in range(len(yp)):
    if yp[j] < 0:
        yp[j] = -1
    else:
        yp[j] = 1
count = 0
for j in range(len(y_test)):
    if yp[j] == y_test[j]:
        count += 1
ac = count / len(y_test)*100

print(ac)

yp = np.zeros(len(y_test))
for i in range(len(y_test)):
    a = np.dot(V_mat, X_test[i, :].T)
    h = 1 / (1 + np.exp(-a))
    yp[i] = np.dot(w_vec, h.T)
for j in range(len(yp)):
    if yp[j] < 0:
        yp[j] = -1
    else:
        yp[j] = 1
count = 0
for j in range(len(y_test)):
    if yp[j] == -1:
        count += 1
bc = count / len(y_test)*100

print(bc)
plt.show()