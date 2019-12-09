import os

def read_data(train_pattern):
    """
    Image features are 321 pixel of each char in word
    :param train_pattern: train dataset directory
    :return: d
    """
    data = []
    files = os.listdir(train_pattern)

    for name in files:
        lines = open(os.path.join(train_pattern, name), 'r')
        label = list(next(lines).strip())
        features = [[int(b) for b in line.split()] for line in lines]
        data.append((label, features))

    return data