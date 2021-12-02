import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms

from matplotlib import pyplot as plt

# Device configuration, use GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters 
num_epochs = 1
# Mini batch size to train before an update
batch_size = 512
learning_rate = 0.0005
learning_rate_decay_rate = 0.99
# Size of input features per example. Carefully picked as 90x170 pixels per image.
# Ideally this can be read from training data but hardcode here and cross
# check with training data.
input_size = 27*44

# Number of frames to use as one sequence for GRU training
# sequence_length = 40
sequence_length = 60

# Number of hidden units in one GRU layer
hidden_size = 256
# Number of GRU layers
num_layers = 5
# num_layers = 5

# Since mouse/gamepad movement is a regression problem and is using MSE loss,
# we want to assign a different weight term to it to balance between
# mouse loss and keyboard loss.
mouse_loss_weight1 = 1.5
mouse_loss_weight2 = 3

# Problem specific parameters
num_keyboard_actions = 7
num_mouse_actions = 2


# Define neural network structure
# Input -> 5 layers GRU -> Linear -> (keyboard) Multi-label Sigmoid
#                          Linear -> (mouse/gamepad) Regression
class MovementNetwork(nn.Module):
    def __init__(self):
        super(MovementNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # Input： tensor of shape (batch_size, sequence_length, feature_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # The final keyboard linear layer that connects GRU layers' output to action space (there are 7 keys in total).
        self.keyboard_linear = nn.Linear(hidden_size, num_keyboard_actions)
        # The final mouse linear layer that regression to mouse X and Y pixel movement.
        self.mouse_linear = nn.Linear(hidden_size, num_mouse_actions)

        
    def forward(self, x):
        # Initialize GRU initial hidden states to zeros.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # Forward propagate GRU layers.
        # gru_out: tensor of shape (batch_size, sequence_length, hidden_size).
        gru_out, _ = self.gru(x, h0)

        # Fetch the hidden state of the last time step.
        gru_out = gru_out[:, -1, :]

        # Forward propagate output layers.
        # Output layer only uses the hidden state of the last time step. 
        keyboard_prediction = self.keyboard_linear(gru_out)
        # keyboard_prediction = nn.sigmoid(keyboard_out)

        # Mouse prediction is the prediction of X & Y value.
        mouse_prediction = self.mouse_linear(gru_out)

        # keyboard_prediction dimension (batch_size, num_keyboard_actions(7))
        # mouse_prediction dimension (batch_size, num_mouse_actions(2))
        return keyboard_prediction, mouse_prediction



if __name__ == "__main__":
    previous_model_path = "movement_network_60s_5l_6.pt"

    model = MovementNetwork().to(device)
    if previous_model_path != '':
        model.load_state_dict(torch.load(previous_model_path), strict = True)

    # Loss and optimizer

    # For keyboard action, this is a multi-label binary classification problem where C=7 (there are 7 keys in total)
    # BCEWithLogitsLoss already contains a sigmoid layer and multi-label support.
    BCE_loss = nn.BCEWithLogitsLoss()

    # For mouse/gamepad action, this is a regression problem to predict the real value of X and Y.
    # Use MSE loss here.
    MSE_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay_rate, verbose=True)

    # Loads training data
    time1 = datetime.datetime.now()
    print("Start loading data")
    X_total = pd.read_csv('data2/train_s1.txt', sep=',', engine='c', header=None, na_filter=False, dtype=np.float64, low_memory=False).to_numpy()
    Y_total = pd.read_csv('data2/label.txt', sep=',', engine='c', header=None, na_filter=False, dtype=np.float64, low_memory=False).to_numpy()

    time2 = datetime.datetime.now()
    data_loading_time = time2 - time1
    print('data loadinng time ', data_loading_time.total_seconds())

    feature_num = X_total.shape[0]
    label_num = Y_total.shape[0]
    feature_size = X_total.shape[1]
    print("feature dimension", X_total.shape)
    print("label dimension", Y_total.shape)

    # Converts training data X into a sliding window for GRU training.
    # Each training example will have dimension (sequence_length, feature_size)
    # X_total will have dimension (num_examples, sequence_length, feature_size)
    X_total = np.lib.stride_tricks.sliding_window_view(X_total, (sequence_length, feature_size))
    X_total = X_total.reshape(feature_num - sequence_length + 1, sequence_length, feature_size)

    # Gets the correspnding labels for each example in X.
    Y_total = Y_total[sequence_length-1: ,:]

    print("feature dimension after transformation ", X_total.shape)
    print("label dimension after transformation ", Y_total.shape)

    # Makes sure the feature size is expected.
    assert feature_size == input_size
    # Makes sure feature and label align.
    assert feature_num == label_num

    losses = []
    # Train the model
    for epoch in range(num_epochs):
        row_counter = 0
        mini_batch_counter = 0
        epoch_start_time = datetime.datetime.now()

        # Shuffle X and Y
        random_indicies = torch.randperm(X_total.shape[0])

        # Minibatch loop
        while row_counter + batch_size < feature_num:
            batch_start_time = datetime.datetime.now()
            X = None
            Y = None

            # Load data for this mini batch
            minibatch_indicies = random_indicies[row_counter:row_counter + batch_size]
            X = X_total[minibatch_indicies]
            Y = Y_total[minibatch_indicies]

            row_counter = row_counter + batch_size

            # The last 2 numbers are mouse/gamepad movement in Y.
            Y_keyboard = Y[: ,:-2]
            Y_mouse = Y[: ,-2:]
            # print("mini batch X shape: ", X.shape)
            # print("mini batch Y_keyboard shape: ", Y_keyboard.shape)
            # print("mini batch Y_mouse shape: ", Y_mouse.shape)

            # Pushes training data and label to tensor on GPU
            X = torch.from_numpy(X).float().to(device)
            Y_keyboard = torch.from_numpy(Y_keyboard).float().to(device)
            Y_mouse = torch.from_numpy(Y_mouse).float().to(device)

            # Forward pass
            keyboard_prediction, mouse_prediction = model(X)
            # print("keyboard_prediction ", keyboard_prediction)
            # print("mouse_prediction ", mouse_prediction)

            # Calculate total loss as total = Loss-keyboard + alpha * Loss-mouse
            keyboard_loss = BCE_loss(keyboard_prediction, Y_keyboard)
            mouse_loss1 = MSE_loss(mouse_prediction[0], Y_mouse[0])
            mouse_loss2 = MSE_loss(mouse_prediction[1], Y_mouse[1])
            total_loss = keyboard_loss + mouse_loss_weight1 * mouse_loss1 + mouse_loss_weight2 * mouse_loss2 

            # print("keyboard_loss ", keyboard_loss)
            # print("mouse_loss ", mouse_loss)
            # print("total_loss ", total_loss)
            # print("epoch", epoch,"mini_batch_counter ", mini_batch_counter)

            # Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            mini_batch_counter = mini_batch_counter + 1

            batch_end_time = datetime.datetime.now()
            mini_batch_time = batch_end_time - batch_start_time
            # print('mini batch time ', mini_batch_time.total_seconds())

        epoch_end_time = datetime.datetime.now()
        epoch_time = epoch_end_time - epoch_start_time
        print('Epoch time ', epoch_time.total_seconds())
        print(f'Epoch {epoch+1}, Loss: {total_loss.item():.8f}')
        losses.append(total_loss.item())
        scheduler.step()

    # Save trained model
    PATH = "movement_network_60s_5l_8.pt"
    torch.save(model.state_dict(), PATH)

    # Plot the loss trend
    plt.plot(losses)
    plt.show()
