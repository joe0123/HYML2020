import sys
import numpy as np
import random
from scipy.ndimage import rotate

seed = 1114
np.random.seed(seed)
random.seed(seed)

def preprocess(image_list):
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = image_list / 255.
    image_list = image_list.astype(np.float32)
    return image_list

if __name__ == "__main__":
    x = np.load(sys.argv[1])
    new_x = []
    for i in range(x.shape[0]): 
        if random.random() > 0.5:
            if random.random() > 0.5:
                new_x.append(rotate(x[i], 15 + random.random() * 30, mode="reflect", reshape=False))
            else:
                new_x.append(rotate(x[i], -15 - random.random() * 30, mode="reflect", reshape=False))
        else:
            new_x.append(np.flip(x[i], axis=1))
    x = np.concatenate((x, new_x), axis=0)
    print(x.shape)
    np.save(sys.argv[2], x)
    
    if len(sys.argv) < 4:
        exit()
    
    y = np.load(sys.argv[3])
    y = np.concatenate((y, y), axis=0)
    print(y.shape)
    np.save(sys.argv[4], y)



