import torch
import pandas as pd
from torch import nn

import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MultiClassModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=0):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def initialise_device():
    # Create device agnostic code
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def scale_data(train, test):
    scaler = StandardScaler()

    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled


def do_mcnn_analysis(df, dims, learning_rate, layers, epochs, scaled=False):
    # Convert class labels to integers and set up training/test data
    classes, route_idx = get_classification_dicts(df)
    df['route_int'] = df['route'].apply(lambda x: route_idx[x])
    x_train, x_test, y_train, y_test = train_test_split(df['grids'], df['route_int'], test_size=0.2, shuffle=False,
                                                        random_state=1)
    if scaled:
        x_train, x_test = scale_data(x_train.tolist(), x_test.list())
    else:
        x_train = x_train.tolist()
        x_test = x_test.tolist()

    # Initialise model
    torch.manual_seed(42)
    device = initialise_device()
    mc_model = MultiClassModel(input_features=dims[0]*dims[1], output_features=len(classes), hidden_units=layers).to(device)

    # Specify loss and optimiser functions
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(mc_model.parameters(), lr=learning_rate)

    # Transfer data to the device
    x_train, y_train = torch.as_tensor(x_train).float().to(device), torch.as_tensor(y_train.to_numpy()).to(device)
    x_test, y_test = torch.as_tensor(x_test).float().to(device), torch.as_tensor(y_test.to_numpy()).to(device)

    # Train model
    for epoch in range(epochs):
        # Training
        mc_model.train()

        # 1. Forward Pass
        y_logits = mc_model(x_train)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        # 2. Calculate loss and energy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # 3. Optimiser zero grad (clear gradients)
        optimiser.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimiser step
        optimiser.step()

        # Testing
        mc_model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = mc_model(x_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            # 2. Calculate test loss and accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        # Show progress summary
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

    # Evaluate model
    mc_model.eval()
    with torch.inference_mode():
        y_logits = mc_model(x_test)

    y_pred_probs = torch.softmax(y_logits, dim=1)
    y_preds = y_pred_probs.argmax(dim=1)

    print(f'Test accuracy: {accuracy_fn(y_true=y_test, y_pred=y_preds):.2f}%')

    print(
        f'Classification report for classifier {mc_model}:\n'
        f'{metrics.classification_report(y_test.cpu(), y_preds.cpu())}\n'
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test.cpu(), y_preds.cpu())
    disp.figure_.suptitle('Confusion Matrix')
    print(f'Confusion Matrix:\n{disp.confusion_matrix}')
    plt.xlabel(classes)
    plt.show()


def get_classification_dicts(df):
    classes = {}
    route_idx = {}

    index = 0
    for route in df['route'].unique().tolist():
        classes[index] = route
        index += 1

    for idx, value in classes.items():
        route_idx[value] = idx

    return classes, route_idx
