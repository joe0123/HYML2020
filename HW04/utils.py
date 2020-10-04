import warnings
warnings.filterwarnings('ignore')

def train_data(path, label):
    if label:
        x = []
        y = []
        with open(path, 'r') as f:
            for line in f.readlines():
                tmp = line.strip().split(' ')
                x.append(tmp[2:])
                y.append(int(tmp[0]))
        return x, y
    else:
        x = []
        with open(path, 'r') as f:
            for line in f.readlines():
                tmp = line.strip().split(' ')
                x.append(tmp)
        return x


def test_data(path):
    x = []
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            tmp = ','.join(line.split(',')[1:]).strip().split(' ')
            x.append(tmp)
    return x


