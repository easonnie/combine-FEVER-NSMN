import numpy as np
from sklearn.utils import murmurhash3_32


# Number feature.
class NumEmbedMatrix(object):
    def __init__(self, size=20000, dim=5, bound=0.1):
        self.matrix = None
        self.dim = dim
        self.size = size
        self.bound = bound

    def create(self, seed=6):
        if self.matrix is None:
            np.random.seed(seed)
            self.matrix = np.random.uniform(-self.bound, self.bound, (self.size, self.dim))

    def clean(self):
        self.matrix = None

    def get_embedding(self, token, seed=6):
        max_length = 5
        if self.matrix is None:
            self.create(seed)

        if len(token) <= max_length and token.isdigit():
            hash_index = murmurhash3_32(token, positive=True) % self.size
            return self.matrix[hash_index]
        else:
            return np.zeros(self.dim)


GlobalNumEmbedding = NumEmbedMatrix()


# encode a list of tokens
def encode_num_in_ltokens(tokens):
    array_list = []
    for token in tokens:
        array_list.append(GlobalNumEmbedding.get_embedding(token))

    # assert len(array_list) == len(tokens)
    num_encoding = np.stack(array_list, axis=0)
    assert len(tokens) == num_encoding.shape[0]
    return num_encoding


if __name__ == '__main__':
    # print(convert_number_feature('11990'))
    # print(convert_number_feature('1190'))
    # print(convert_number_feature('124124'))
    # print(convert_number_feature('1990'))
    # print(convert_number_feature('fwef'))
    # print(convert_number_feature('2412.23'))
    print(encode_num_in_ltokens(['aab', '1924', '1984', '1988', '1924']))
    print(encode_num_in_ltokens(['aab', '1924', '1984', '1988', '1924']))
    print(encode_num_in_ltokens(['aab', '1924', '1984', '1988', '1924']) ** 2)