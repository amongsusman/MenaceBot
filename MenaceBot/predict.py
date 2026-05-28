import numpy as np
from chess import Board, pgn
from auxfuncs import convert
import pickle
import torch
from model import ChessModel

def convert_input(board: Board):
    new_matrix = convert(board)
    X_tensor = torch.tensor(new_matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor

with open("models/move_to_int.pkl", "rb") as file:
    move_to_int = pickle.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chess_model = ChessModel(num_classes=len(move_to_int))
chess_model.load_state_dict(torch.load("models/Menace1.pth"))
chess_model.to(device)
chess_model.eval()

int_to_move = {v: k for k, v in move_to_int.items()}

def predict_move(board: Board):
    X_tensor = convert_input(board).to(device)
    with torch.no_grad():
        logits = chess_model(X_tensor)
    
    logits = logits.squeeze(0)
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probs)[::-1]

    for index in sorted_indices:
        move = int_to_move[index]
        if move in legal_moves_uci:
            return move
        
    return None

board = Board()
while True:
    move = input("put ur move ")
    board.push_uci(move)
    print(board.unicode())
    print()
    ai_move = predict_move(board)
    board.push_uci(ai_move)
    print(board.unicode())


