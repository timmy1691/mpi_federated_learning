import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.stats import ortho_group
from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("datasets/adult_host.csv", index_col="id")
label_data = pd.read_csv("datasets/adult_guest.csv", index_col="id")
target = label_data["y"]

num_rows, num_cols = data.shape

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

ortho_trans = []
num_trans = 10
for i in range(num_trans):
    ortho_trans.append(ortho_group.rvs(num_cols))

combined_training_data = X_train
trans_target = y_train
combined_testing_data = X_test
trans_test = y_test
for trans in ortho_trans:
    transformed_data = X_train@trans
    trans_test_data = X_test@trans
    combined_training_data = np.concatenate((combined_training_data, transformed_data), axis=0)
    combined_testing_data = np.concatenate((combined_testing_data, trans_test_data), axis=0)
    trans_test = np.concatenate((trans_test, y_test), axis=0)
    trans_target = np.concatenate((trans_target, y_train), axis=0)


class NeuralNetwork(nn.Module):
    def __init__(self, input, output, hidden_size=[100, 100]):
        super().__init__()
        self.input_size = input
        self.output_size = output
        self.hidden_size = hidden_size
        self.model = self.create_model()

    def create_model(self):
        layers = []
        for i, _ in enumerate(self.hidden_size):
            if i == 0:
                layers.append(nn.Linear(self.input_size, self.hidden_size[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(self.hidden_size[i-1], self.hidden_size[i]))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_size[i], self.output_size))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)


    def forward(self, x):
        logits = self.model(x)
        return logits
    
og_loaded_data = DataLoader(list(zip(combined_training_data, trans_target.reshape(-1,1))), batch_size=200, shuffle=True)
og_test_data = DataLoader(list(zip(combined_testing_data, trans_test.reshape(-1,1))), batch_size=200, shuffle=True)


model = NeuralNetwork(num_cols, 1, hidden_size=[100, 100, 100])
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Train the net
n_epochs = 10000
loss_per_batch = []
total_loss = []
for epoch in tqdm(range(n_epochs)):
        running_loss = 0
        for i, (inputs, labels) in enumerate(og_loaded_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs.float())
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()

        if epoch % 100 == 0:
            total_loss.append(running_loss)
            loss_per_batch.append(running_loss / (i + 1))


        if epoch % 2000 == 0:
            print(f"epoch {epoch} : loss {running_loss/(i+1)}")

# torch.save(model.state_dict(), "base_line_model_parameters")

accuracy = 0
for j, (test_input, test_label) in enumerate(og_test_data):
    test_input = test_input.to(device)
    test_label = test_label.to(device)
    test_predict = model(test_input.float())
    rounded = np.round(test_predict.cpu().detach().numpy())
    accuracy += accuracy_score(rounded, test_label.cpu())

results = {"number_of_transformations" : num_trans,
           "accuracy" : accuracy}

results_df = pd.DataFrame(results, index=False)

results_df.to_csv("orthogonality_data_testing.csv")
