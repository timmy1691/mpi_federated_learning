# import subprocess
# import time

# start_time = time.time()
# subprocess.run(["mpiexec", "-n", "4" , "python", "ortho_multi_party_simple_version.py"])
# end_time = time.time()

# print("time is ", end_time - start_time)


import pandas as pd

pd.read_csv("results.csv")