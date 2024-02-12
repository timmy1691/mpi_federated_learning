import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
import time
import sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

hidden_dim = None
try:
    n_rows = int(sys.argv[1])
    n_cols = int(sys.argv[2])
    hidden_dim = int(sys.argc[3])
except Exception :
    data_path = sys.argv[0]
    hidden_dim = int(sys.argv[2])

if hidden_dim == None:
    hidden_dim = n_cols


data = np.random.uniform(0,1,size=(n_rows,n_cols))
torch.manual_seed(42)
X_tensor = torch.FloatTensor(data)

class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size, enc_hidden_size = None, dec_hidden_size=None):
        super(Autoencoder, self).__init__()
        if enc_hidden_size == None:
            enc_hidden_size = input_size/3
        if dec_hidden_size == None:
            dec_hidden_size = enc_hidden_size        
        self.encoder = nn.Sequential(
                            nn.Linear(input_size, enc_hidden_size),
                            nn.ReLU(),
                            nn.Linear(enc_hidden_size, encoding_size),
                            nn.ReLU()
                        )
        
        self.decoder = nn.Sequential(
                            nn.Linear(encoding_size, dec_hidden_size),
                            nn.ReLU(),
                            nn.Linear(dec_hidden_size, input_size),
                            nn.ReLU()
                        )
        
    def forward(self, x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x
    
def train(model, input_data, error_value = 0.1, max_epoch = 100000):
        # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(max_epoch):
    # Forward pass
        outputs = model(input_data)
        loss = criterion(outputs, input_data)
        # break early
        if loss.item() < error_value:
            break
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss for each epoch
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


#####
if rank == 0:
    target = np.random.randint(0,2,size=(n_rows,1))
    training_start = time.time()


print(f"party {rank} initializing autoencoders")

if rank == 0:
    # Setting random seed for reproducibility
    input_size = data.shape[1]  # Number of input features
    encoding_dim = hidden_dim  # Desired number of output dimensions
    model = Autoencoder(input_size, encoding_dim)

    # # Loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.003)

    print("start AE training")
    train(model, X_tensor)
    # Encoding the data using the trained autoencoder
    encoded_data = model.encoder(X_tensor).detach().numpy()
    print("receiving data")
    encoded_data_rec = comm.recv(source=1, tag=11)
    full_encoded_data = np.concatenate((encoded_data, encoded_data_rec), axis=1)

    learning_model = LogisticRegression()
    learning_model.fit(full_encoded_data, target.ravel())

    training_finish_time = time.time()

    print("total time ", training_finish_time - training_start)
elif rank == 1:

    torch.manual_seed(42)

    input_size = data.shape[1]  # Number of input features
    encoding_dim = 50  # Desired number of output dimensions
    model = Autoencoder(input_size, encoding_dim)

    train(model, X_tensor)

    # Encoding the data using the trained autoencoder
    
    encoded_data = model.encoder(X_tensor).detach().numpy()
    print("start sending data")
    comm.send(encoded_data, dest=0, tag = 11)



