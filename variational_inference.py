from io_data import read_data, write_data
import numpy as np
import scipy.stats
import math

num_iter = 10
J = 0.5
sigma = 0.5
filenumber = 1
L1 = scipy.stats.norm(1, sigma)
L_1 = scipy.stats.norm(-1, sigma)


def transform_data_shape(data):
    """transform data to 2d-array so we can access pixels with [i,j]"""
    new_data = np.zeros((int(data[-1, 0])+1, int(data[-1, 1])+1))
    for row in data:
        new_data[int(row[0]), int(row[1])] = row[2]
    return new_data


def transform_back_shape(data):
    """put it back to original shape"""
    new_data = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            new_data += [[i, j, data[i,j]]]
    return np.array(new_data)


def transform_data_values(data):
    """[0,255] to [-1,1]"""
    data /= 255
    data -= 0.5
    data *= 2
    return data


def transform_back_values(data):
    """[1,1] to [0,255]"""
    data /= 2
    data += 0.5
    data *= 255
    return data


def find_neighbors(i, j, i_bound, j_bound):
    """find de neighbors of pixel (i,j) in a square of size squaresize"""
    neighbors = []
    if i == 0:
        if j == 0:
            neighbors += [[i, j+1], [i+1, j]]
        elif j == j_bound:
            neighbors += [[i, j - 1], [i + 1, j]]
        else:
            neighbors += [[i, j - 1], [i + 1, j], [i, j+1]]
    elif i == i_bound:
        if j == 0:
            neighbors += [[i, j+1], [i-1, j]]
        elif j == j_bound:
            neighbors += [[i, j - 1], [i - 1, j]]
        else:
            neighbors += [[i, j - 1], [i, j + 1], [i - 1, j]]
    else:
        if j == 0:
            neighbors += [[i, j + 1], [i + 1, j], [i - 1, j]]
        elif j == j_bound:
            neighbors += [[i, j - 1], [i + 1, j], [i - 1, j]]
        else:
            neighbors += [[i, j - 1], [i + 1, j], [i - 1, j], [i, j + 1]]
    return neighbors


def sum_neighbors(neighbors, data):
    sum = 0
    for neighbor in neighbors:
        #we compute the mean field influence with all Wij=1
        sum += data[neighbor[0], neighbor[1]]
    return sum


def qi(i, j, x, data, mu):
    mean_field = sum_neighbors(find_neighbors(i,j, data.shape[0]-1, data.shape[1]-1), mu)
    if data[i, j] == 1:
        L = L1
    else:
        L = L_1
    return math.exp(x*J*mean_field + L.pdf(x))/(math.exp(x*J*mean_field + L.pdf(x))+math.exp(-x*J*mean_field + L.pdf(-x)))


def compute_entropy(data, mu):
    entropy = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            entropy += qi(i, j, 1, data, mu)*math.log(qi(i, j, 1, data, mu))
    return -entropy/(data.shape[0]*data.shape[1])


def update_mu(i, j, data, mu):
    mean_field = sum_neighbors(find_neighbors(i, j, data.shape[0]-1, data.shape[1]-1), mu)
    if data[i, j] == 1:
        L = L1
    else:
        L = L_1
    mu[i, j] = math.tanh(J*mean_field + 0.5*(L.pdf(1)-L.pdf(-1)))
    return mu


def variational_inference(data):
    entropy = 0
    new_entropy = 1
    mus = np.zeros(data.shape)
    # initialize mu
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 1:
                mus[i, j] = L1.pdf(1) - L1.pdf(-1)
            else:
                mus[i, j] = L_1.pdf(1) - L_1.pdf(-1)
    entropy = compute_entropy(data, mus)
    iter = 0
    print("initialized")
    while abs(entropy-new_entropy) > 0.0001:
        iter += 1
        entropy = new_entropy
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                mus = update_mu(i, j, data, mus)
        new_entropy = compute_entropy(data, mus)
        print(new_entropy)
        iter_mus = mus.copy()
        denoized_data = np.round(iter_mus)
        denoized_data = transform_back_values(denoized_data)
        denoized_data = transform_back_shape(denoized_data)
        write_data(denoized_data,
                   str(J) + '_J_' + str(sigma) + '_sigma_' + str(iter) + '_epoch' + str(filenumber) + '_noise_vi.txt')
        read_data(
                   str(J) + '_J_' + str(sigma) + '_sigma_' + str(iter) + '_epoch' + str(filenumber) + '_noise_vi.txt', True, False, True)

    return mus


def draw_new_samples(data):
    new_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if ((data[i, j]+1)/2) > np.random.rand():
                new_data[i,j] = 1
    return new_data


for filenumber in range(1,5):
    data, image = read_data('../a1/'+str(filenumber)+'_noise.txt', True)
    data = transform_data_shape(data)
    # put data in {-1, 1} space
    data = transform_data_values(data)
    variational_inference(data)