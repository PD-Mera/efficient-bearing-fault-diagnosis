from scipy.io import loadmat
from os.path import join
from os import listdir
import numpy as np
from math import floor
from PIL import Image

ROOT = './Bearing-Dataset-16x16-noise'
NAME_1 = ['train', 'test']
NAME_2 = ['B', 'I', 'L', 'N', 'O']

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

def filter_1d(input_array: np.ndarray, 
              kernel: np.ndarray,
              stride: int,
              padding: int = 0):
    results = []
    if padding != 0:
        pad = np.zeros(padding)
        input_array = np.concatenate((pad, input_array, pad), axis=None)
    M = len(input_array)
    N = len(kernel)
    step = floor((M - N + 2 * padding) / stride + 1)
    
    for i in range(step):
        local_array = input_array[i:i+N]
        if len(local_array) < N:
            break
        ew_array = np.multiply(local_array, kernel)
        results.append(np.average(ew_array))
        
    
    return np.array(results)

if __name__ == "__main__":
    kernel = np.ones(9)
    for name_1 in NAME_1:
        for name_2 in NAME_2:
            folder_link = join(ROOT, name_1, name_2)
            for mat_file in listdir(folder_link):
                if is_mat_file(mat_file):
                    mat_link = join(folder_link, mat_file)
                    decode_dict = loadmat(mat_link)
                    temp = decode_dict['temp'].reshape(1024)
                    temp = filter_1d(temp, kernel, stride = 4, padding = 2)
                    np.save(mat_link[:-3] + 'npy', temp)
                    print(temp.shape)
