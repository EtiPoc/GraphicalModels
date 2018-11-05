from io_data import read_data, write_data
import numpy as np
from scipy.cluster.vq import kmeans2
import scipy.stats
import math
import cv2

def k_means_initialization(data, K):
    means, labels = kmeans2(data[:, [2, 3, 4]], K)
    stds = [np.cov(data[labels == k][:, [2, 3, 4]].T) for k in range(K)]
    pis = [sum(labels == k)/data.shape[0] for k in range(K)]
    return means, stds, pis


def compute_responsibilities(data, means, stds, pis, K):
    responsibilities = np.zeros((data.shape[0], K))
    for i in range(data.shape[0]):
        for k in range(K):
            responsibilities[i, k] = pis[k] * scipy.stats.multivariate_normal.pdf(data[i, [2, 3, 4]], mean=means[k], cov=stds[k], allow_singular=True)
    responsibilities = np.array(responsibilities)
    for i in range(data.shape[0]):
        responsibilities[i] = responsibilities[i]/sum(responsibilities[i])
    return responsibilities


def update_means(data, responsibilities, K, N):
    product = [[list(responsibilities[i, j] * data[i, [2, 3, 4]]) for j in range(K)] for i in range(data.shape[0])]
    product = np.array(product)
    means = [sum(product[:, k])/N[k] for k in range(K)]
    return means


def update_stds(data, responsibilities, means, K, N):
    sums = np.zeros((K, 3, 3))
    for i in range(data.shape[0]):
        for k in range(K):
        # print(data[i, [2, 3, 4]] - means[0], data[i, [2, 3, 4]] - means[1])
        # print(np.sqrt(responsibilities[i, 0]), np.sqrt(responsibilities[i, 1]))
        #     print(responsibilities[i, k], data[i, [2, 3, 4]])
            error = (data[i, [2, 3, 4]] - means[k]).reshape(3,1)
            product = np.dot(error, error.T)
            # print(product)
            sums[k] += responsibilities[i, k] * product
    stds = [sums[k]/N[k] for k in range(K)]
    return stds


def update_pis(N, K):
    pis = [N[k]/sum(N) for k in range(K)]
    return pis


def compute_likelihood(data, means, stds, pis, K):
    likelihood = 0
    for i in range(data.shape[0]):
        likelihood_i = sum([pis[j] * scipy.stats.multivariate_normal.pdf(data[i, [2, 3, 4]], mean=means[j], cov=stds[j], allow_singular=True) for j in range(K)])
        # print(likelihood_i)
        likelihood += math.log(likelihood_i)
    return likelihood


def em(data, K=2, threshold = 1):
    means, stds, pis = k_means_initialization(data, K)
    likelihood = 0
    new_likelihood = 2
    responsibilities = np.zeros(data.shape)
    while (likelihood - new_likelihood)**2 > threshold:
        likelihood = new_likelihood
        #E-ste
        responsibilities = compute_responsibilities(data, means, stds, pis, K)
        N = sum(responsibilities)
        print(N)
        #M-step
        means = update_means(data, responsibilities, K, N)
        stds = update_stds(data, responsibilities, means, K, N)
        pis = update_pis(N, K)
        new_likelihood = compute_likelihood(data, means, stds, pis, K)
        print(new_likelihood)
    return responsibilities


def process_save(animal, data, responsibilities, k):
    max_responsibilities = [np.argmax(responsibilities[i]) for i in range(responsibilities.shape[0])]
    data_classified = np.zeros(data.shape)
    for i in range(data.shape[0]):
        data_classified[i, 0] = data[i, 0]
        data_classified[i, 1] = data[i, 1]
        data_classified[i, 2] = (100/(k-1)) * max_responsibilities[i]
        data_classified[i, 3] = 0
        data_classified[i, 4] = 0
    write_data(data_classified, animal+'_mask.txt')
    for k in range(k):
        for i in range(data.shape[0]):
            data_classified[i, 2] = data[i, 2] * (max_responsibilities[i] == k)
            data_classified[i, 3] = data[i, 3] * (max_responsibilities[i] == k)
            data_classified[i, 4] = data[i, 4] * (max_responsibilities[i] == k)
        write_data(data_classified, animal+'_seg'+str(k)+'.txt')


def add_index(img):
    copy = np.zeros((img.shape[0], img.shape[1], 5))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            copy[i, j, 0] = i
            copy[i, j, 1] = j
            copy[i, j, 2] = img[i, j, 0]
            copy[i, j, 3] = img[i, j, 1]
            copy[i, j, 4] = img[i, j, 2]
    copy = copy.reshape((copy.shape[0]*copy.shape[1], 5))
    for i in range(copy.shape[0]):
        copy[i, 2] = 100 * copy[i, 2] / 255
        copy[i, 3] = copy[i, 3] - 128
        copy[i, 4] = copy[i, 4] - 128
    return copy


# for animal in ['cow', 'owl', 'zebra', 'fox']:
#     print(animal)
#     data, image = read_data("../a2/"+animal+".txt", False)
#     print('start em')
#     responsibilities = em(data, 2, threshold=0.0000001)
#     print("process")
#     process_save(animal, data, responsibilities, 2)


for image in ['pogba','marseille']:
    img = cv2.imread(image+'.jpg')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    copy = add_index(img2)
    write_data(copy, image+'.txt')
    k=2
    if image=='marseille':
        k=3
    data, useless = read_data(image+".txt", False)
    responsibilities = em(data, k)
    process_save(image, data, responsibilities, k)










