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

#add games
games = []

dir_path = Path("/data/pgn")

for file in dir_path:
    file_game = chess.pgn.read_game(file)
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
model = ChessModel(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#train AI
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loss = criterion(outputs, labels)
    running_loss += loss.item()






