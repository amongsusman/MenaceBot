import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from auxfuncs import create_game_inputs, encode_moves
from pathlib import Path
from model import ChessModel
from dataset import ChessDataset
from tqdm import tqdm
import chess.pgn

#add games
games = []

dir_path = r"C:\\Users\\compsci\\Downloads\\MenaceBot\\data\\pgn"

for entry in os.scandir(dir_path):
    if entry.is_file():
        with open(entry, "r") as file:
            while True:
                file_game = chess.pgn.read_game(file)
                if file_game is None:
                    break
                games.append(file_game)

#make tensors
X, y = create_game_inputs(games)

X = X[0:2500000]
y = y[0:2500000]

y, move_to_int = encode_moves(y)

num_classes = len(move_to_int)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

#make dataset, dataloader, and model 
dataset = ChessDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#train AI
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(running_loss / len(dataloader))
torch.save(model.state_dict(), "/models/Menace1.pth")




