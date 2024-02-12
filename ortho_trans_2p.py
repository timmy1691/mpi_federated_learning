import numpy as np
from phe import paillier
from mpi4py import MPI
from tqdm import tqdm
import time
from scipy.stats import ortho_group
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys

hidden_dim = None
try:
    n_rows = int(sys.argv[1])
    n_cols = int(sys.argv[2])
    data = np.random.uniform(0,1,size=(n_rows,n_cols))
except Exception :
    data_path = sys.argv[0]
    data = pd.read_csv(data_path)

aggregation_time = []
full_ete_training_time = []

num_iterations = 1000
for i in range(num_iterations):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # print("iterations ")
    start_time = time.time()
    # rank 0  being the active party

    if rank == 0:
        print(f"iteration {i}")
        target = np.random.randint(0, 2, size=(n_rows,1))
        comm.send(n_cols, dest=1, tag=0)
    elif rank == 1:
        other_dim = comm.recv(source=0, tag=0)

    if rank == 1:
        transformation = ortho_group.rvs(other_dim + n_cols)
        trans_0 = transformation[:n_cols]
        trans_1 = transformation[n_cols:]
        comm.send(trans_1, dest=0, tag=1)   
        trans_data = data@trans_0
        comm.send(trans_data, dest=0, tag=2)

    elif rank == 0:
        transformation = comm.recv(source=1, tag=1)
        # print("receiving transformation")
        received_data = comm.recv(source=1, tag=2)
        # print("receiving data")
        trans_data = data@transformation
        total_data = trans_data + received_data
        end_time = time.time()
        aggregation_time.append(end_time - start_time)

        model = LogisticRegression()
        model.fit(total_data, target.ravel())
        full_end_time = time.time()
        full_ete_training_time.append(full_end_time - start_time)
if rank == 0:
    print("Average time for a matrix ", sum(aggregation_time)/len(aggregation_time))
    print("full ete training ", sum(full_ete_training_time)/ len(full_ete_training_time))
if rank == 0:
    try:
        print("average aggregation time : ", sum(aggregation_time)/len(aggregation_time))
        temp_df = pd.read_csv("results.csv")
        temp_df["2_party_ete_add_time"] = aggregation_time
        temp_df.to_csv("results.csv", index=False)
    except pd.errors.EmptyDataError:
        temp_df = pd.DataFrame({"2_party_ete_add_time" : aggregation_time})
        temp_df.to_csv("results.csv", index=False)