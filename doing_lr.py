import numpy as np
from phe import paillier
from mpi4py import MPI
from tqdm import tqdm
import time
import pandas as pd

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def sig_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def entropy(y, y_pred):
    return y*np.log(y_pred) + (1-y)*np.log(1-y_pred)

def entropy_deriv(y, y_pred):
    return y/y_pred + (1-y)/(1-y_pred)

total_training_time = []
avg_forward_time = []
avg_backward_time = []

number_trials = 1
for trial in range(number_trials):
    print(f"starting trial {trial}")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print("total parties : ", size)

    data = np.random.uniform(0,1,size=(1000,10)).astype(np.float64)

    num_rows, num_cols = data.shape
    local_params = np.random.uniform(-1/np.sqrt(num_cols), 1/np.sqrt(num_cols), size=(1,num_cols)).astype(np.float64).T
    enc_local_params = np.zeros_like(local_params).astype("O")

    public_key = None
    private_key = None

    intermediate = np.zeros((1000,1)).astype(np.float64)
    total_intermediate = np.zeros_like(intermediate).astype(np.float64)
    enc_grad = np.zeros(1).astype("O")

    recv_buffer = np.zeros(1).astype("O")

    lr = 0.01
    if rank == 0:
        training_start_time = time.time()
        forward_time = []
        backward_time = []

    # communicating the keys
    if rank == 0:
        print("sending key")
        public_key, private_key = paillier.generate_paillier_keypair()
        comm.send(public_key, dest=1, tag=11)
    elif rank ==1 :
        public_key = comm.recv( source=0, tag=11)
        print("key received ", public_key)

        # public_key_store[0] = recv_buffer[0]

    if rank == 0:
        target = np.random.randint(2 ,size=(1000,1))
    elif rank == 1:
        enc_start_time = time.time()
        print("encrypting the weights")
        for i, param in tqdm(enumerate(local_params)):
            enc_local_params[i,0] = public_key.encrypt(param[0])
        enc_end_time = time.time()
        print("finished encrypting weights")

    print(f"party {rank} starting  model training")
    iterations = 100
    # print(f"party {rank} beginning training")
    for t in range(iterations):
        print(f"starting iteration {t}")
        # communicating intermediate results
        if rank == 0:
            forward_time_start = time.time()

        if rank == 0:
            total_intermediate = data@local_params
            intermediate = comm.recv(source=1, tag=2*iterations+t)
            print("decrypting the intermediate results")
            for i, inter in tqdm(enumerate(intermediate)):
                total_intermediate[i][0] += private_key.decrypt(intermediate[i][0])
        else:
            print("performing dot product with encrypted weights")
            intermediate = data@enc_local_params
            comm.send(intermediate, dest=0, tag=2*iterations+t)

        if rank == 0:

            forward_time_end = time.time()
            forward_time.append(forward_time_end - forward_time_start)
            
        one_vec = np.ones((num_rows, 1))


        if rank == 0:
            final_output = sigmoid(total_intermediate)
            # print(final_output)

        # calculating loss and back-propagation
        print("start of back_propagation")
        if rank == 0:
            back_start_time = time.time()
            loss = entropy(target, final_output)
            loss_grad = target - final_output
            # print(loss_grad.shape)
            # print(target.shape)
            # print(final_output.shape)
            one_vec = np.ones_like(loss_grad)
            agg_loss_grad = sum(loss_grad)/len(loss_grad)
            sig_grad = sig_derivative(total_intermediate)
            enc_grad = np.zeros_like(sig_grad).astype("O")
            final_loss = sig_grad*agg_loss_grad

            print("start encrypting the gradients")
            for i, grad in tqdm(enumerate(sig_grad)):
                enc_grad[i] = public_key.encrypt(final_loss[i][0])

            comm.send(enc_grad, dest=1, tag=3*iterations+t)
            print(f"party {rank} start updating params")
            # batched_grads = one_vec.T@sig_grad
            for index in range(num_rows):
                local_params -= (lr*data[index]*sig_grad[index]*agg_loss_grad).reshape(-1,1)

        else:
            print(f"party {rank} start updating params")
            enc_grad = comm.recv(source=0, tag=3*iterations+t)
            # batched_enc_grads = one_vec.T@enc_grad
            # print("gradient shape ", enc_grad.shape)
            # print("data shape ", data.shape)

            for index in range(num_rows):
                # print("local params shape ", enc_local_params.shape)
                # print("enc_grad_shape ", enc_grad[index].shape)
                # print("data shape ", data[index].shape)
                # print("combined thing shape ", (data[index]*enc_grad[index]).reshape(-1,1).shape)
                enc_local_params -= lr*(data[index]*enc_grad[index]).reshape(-1,1)
            # print(f"process {rank}, params {enc_local_params}")
        
        if rank == 0:
            back_end_time = time.time()
            backward_time.append(back_end_time - back_start_time)

    training_end_time = time.time()
    if rank == 0:
        total_training_time.append(training_end_time - training_start_time)
        avg_forward_time.append(sum(forward_time)/len(forward_time))
        avg_backward_time.append(sum(backward_time)/len(backward_time))
    # if rank == 0:
    #     print(f"process {rank} average forward time ", sum(forward_time)/len(forward_time))
    #     print(f"process {rank} average forward time ", sum(forward_time)/len(forward_time))

    # print(f"total training time for {iterations} iterations is", training_end_time - training_start_time)\

if rank == 0:
    try:
        lr_results = pd.read_csv("logistic_reg_timing.csv")
        lr_results[f"dim_{num_cols}_party_{size}_time"] =  total_training_time
        lr_results[f"dim_{num_cols}_party_{size}_forward_time"] = avg_forward_time
        lr_results[f"dim_{num_cols}_party_{size}_backward_time"] = avg_backward_time
        lr_results.to_csv("logistic_reg_timing.csv", index=False)
    except pd.errors.EmptyDataError:
        data_dict = {f"dim_{num_cols}_party_{size}_time" : total_training_time,
                     f"dim_{num_cols}_party_{size}_forward_time" : avg_forward_time,
                     f"dim_{num_cols}_party_{size}_backward_time" : avg_backward_time}
        
        lr_results = pd.DataFrame(data_dict)
        lr_results.to_csv("logistic_reg_timing.csv", index=False)