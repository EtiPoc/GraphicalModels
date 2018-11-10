from io_data import read_data, write_data
import numpy as np
import scipy.stats
import math

num_iter = 20
J = 0.5
sigma = 1


def transform_data_shape(data):
    square_size = int(data[-1, 0])+1
    new_data = np.zeros((square_size, square_size))
    for row in data:
        new_data[int(row[0]), int(row[1])] = row[2]
    return new_data


def transform_back_shape(data):
    new_data = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            new_data += [[i, j, data[i,j]]]
    return np.array(new_data)


def transform_data_values(data):
    data /= 255
    data -= 0.5
    data *= 2
    return data


def transform_back_values(data):
    data /= 2
    data += 0.5
    data *= 255
    return data


def find_neighbors(i, j, square_size):
    square_size = square_size-1
    neighbors = []
    if i == 0:
        if j==0:
            neighbors += [[i, j+1], [i+1, j]]
        elif j == square_size:
            neighbors += [[i, j - 1], [i + 1, j]]
        else :
            neighbors += [[i, j - 1], [i + 1, j], [i, j+1]]
    elif i == square_size:
        if j==0:
            neighbors += [[i, j+1], [i-1, j]]
        elif j == square_size:
            neighbors += [[i, j - 1], [i - 1, j]]
        else :
            neighbors += [[i, j - 1], [i, j + 1], [i - 1, j]]
    else:
        if j == 0:
            neighbors += [[i, j + 1], [i + 1, j], [i - 1, j]]
        elif j == square_size:
            neighbors += [[i, j - 1], [i + 1, j], [i - 1, j]]
        else:
            neighbors += [[i, j - 1], [i + 1, j], [i - 1, j], [i, j + 1]]
    return neighbors




def sum_neighbors(neighbors, data):
    sum = 0
    for neighbor in neighbors:
        sum += data[neighbor[0], neighbor[1]]
    return sum


def gibbs_sampling(data):
    observations = data
    square_size = observations.shape[0]
    for t in range(num_iter):
        print('epoch ', t)
        for i in range(square_size):
            # print(' line ', i)
            for j in range(square_size):
                y = observations[i, j]
                # print(y)
                local_evidence_law = scipy.stats.norm(y, sigma)

                neighbors = find_neighbors(i, j, square_size)
                sum_neigh = sum_neighbors(neighbors, data)
                # print(J, sum_neigh)
                term_1 = local_evidence_law.pdf(1) * math.exp(J*sum_neigh)
                term_minus_1 = local_evidence_law.pdf(-1) * math.exp(- J * sum_neigh)

                proba_x_1 = term_1/(term_1+term_minus_1)
                rand = np.random.rand()
                if rand < proba_x_1:
                    observations[i, j] = 1
                else:
                    observations[i, j] = -1
        if (t%5) ==0:
            epoch = observations.copy()
            denoized_data = transform_back_values(epoch)
            denoized_data = transform_back_shape(denoized_data)
            write_data(denoized_data, str(J)+'_J_'+str(sigma)+'_sigma_'+ str(t)+'_epoch'+str(filenumber)+'_noise.txt')
            read_data(str(t)+'_epoch.txt', True, True)
    return observations


def denoise_gibbs(filenumber):
    data, image = read_data(str(filenumber)+'_noise.txt', True)
    data = transform_data_shape(data)
    # put data in {-1, 1} space
    data = transform_data_values(data)

    denoized_data = gibbs_sampling(data)

    denoized_data = transform_back_values(denoized_data)
    denoized_data = transform_back_shape(denoized_data)
    write_data(denoized_data, 'final_denoised'+str(filenumber)+'_noise.txt')
    read_data('final_denoised'+str(J)+'_J_'+str(sigma)+'_sigma_' + str(filenumber)+'_noise.txt', True, True)
    return denoized_data


for filenumber in range(1,5):
    #     for sigma in [0.5, 1, 2]:
    #         for J in [0.5, 1, 2]:
    denoise_gibbs(filenumber)


