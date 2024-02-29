from multiprocessing import Pool
import numpy as np
from phe import paillier
import time

public_key, private_key = paillier.generate_paillier_keypair()

def encryption( value):
    return public_key.encrypt(value)

if __name__ == '__main__':

    start_multi_time = time.time()
    num_cols = 100
    local_params = np.random.uniform(-1/np.sqrt(num_cols), 1/np.sqrt(num_cols), size=(num_cols)).astype(np.float64).T
    with Pool(3) as p:
        enc_local_params = p.map(encryption, local_params.tolist())
    p.close()

    end_multi_time = time.time()
    print("multi-paral time ", end_multi_time - start_multi_time)

    # enc_params = np.zeros_like(local_params).astype("O")
    # start_sequential_time = time.time()
    # for i , param in enumerate(local_params):
    #     enc_params[i] = public_key.encrypt(local_params[i])
    # end_seq_time = time.time()

    # print("sequential time ", end_seq_time - start_sequential_time)

    data = np.random.uniform(-1,1,size=(100, 100))

    for thing in zip(data, np.array([enc_local_params])):
        print(thing[0].shape)
        print(thing[1].shape)
        thing[0] @ thing[1]
        break