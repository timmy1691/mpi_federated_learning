import numpy as np
from mpi4py import MPI
from tqdm import tqdm
import time
from scipy.stats import ortho_group
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

iterations = 1000

if rank == 0:
    print("total number of parties ", size)
    multi_party_simp_agg_time = []

for i in tqdm(range(iterations)):

    if rank == 0:
        start_time = time.time()

    data = np.random.uniform(0, 1, size=(1000, 50))
    num_rows , num_cols = data.shape

    # number 0 is active party
    transformation = ortho_group.rvs(num_cols)
    transformed_data = data@transformation

    if rank == 0:
        for index in range(1, size):
            received_thing = comm.recv(source=index, tag=1)
            transformed_data = np.concatenate((transformed_data, received_thing))
    else:
        comm.send(transformed_data, dest=0, tag=1)

    if rank == 0:
        end_time = time.time()
        multi_party_simp_agg_time.append(end_time - start_time)

if rank == 0:
    try:
        print(f"average for {size} parties aggregation time : ", sum(multi_party_simp_agg_time)/len(multi_party_simp_agg_time))
        temp_df = pd.read_csv("results.csv")
        temp_df[f"{size}_party_ete_agg_time"] = multi_party_simp_agg_time
        temp_df.to_csv("results.csv", index=False)
    except pd.errors.EmptyDataError:
        temp_df = pd.DataFrame({f"{size}_party_ete_agg_time" : multi_party_simp_agg_time})
        temp_df.to_csv("results.csv", index=False)
