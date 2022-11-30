DATASET_ROOT_DIR = '/root/workspace_2/Bearing-Dataset-16x16-noise'
PHASE = ['train', 'test']
CLASSES = {'B': [0, ''], 
           'I': [1, ''], 
           'L': [2, ''], 
           'N': [3, ''], 
           'O': [4, '']}

TRAINING_EPOCH = 6
BATCH_SIZE = 16

if __name__ == '__main__':
    from os import listdir
    from os.path import join

    print(listdir(join(DATASET_ROOT_DIR, PHASE[0]))) 