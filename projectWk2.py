import numpy as np
import random

updated_board = []

def create_board():
    board = np.zeros((3,3), dtype=int)
    return board

def place(board, player, position):
    global updated_board
    board[position[0]][position[1]] = player
    updated_board = board

def posibilities(board):
    #b = board
    pos = np.where(board==0)
    return list(zip(pos[0], pos[1]))
    #return pos

def random_place(board, player):
    position = random.choice(posibilities(board))
    #print("player %s position is %s" %(str(player), str(position)))
    place(board, player, position)

def row_win(board, player):
    for row in board:
        nrow = np.array(row)
        if np.all(nrow==player):
            return True
    return False

def col_win(board, player):
    b = np.all(board==player, axis=0)
    if True in b:
        return True
    else:
        return False

def diag_win(board, player):
    if np.all(board.diagonal()==player):
        return True
    elif np.all(np.fliplr(board).diagonal()==player):
        return True
    else:
        return False

def evaluate(board):
    winner = 0
    for player in [1,2]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player
            return winner
    if np.all(board !=0) and winner == 0:
        winner = -1
    return winner

def play_game():
    board = create_board()
    player = 1
    while(0 in board):
        if player == 2:
            random_place(board, player)
            player = 1
            if evaluate(board):
                return evaluate(board)
        elif player == 1:

            random_place(board, player)
            player = 2
            if evaluate(board):
                return evaluate(board)

def play_strategic_game():
    board = create_board()
    player = 1
    while(0 in board):
        if player == 2:
            random_place(board, player)
            player = 1
            if evaluate(board):
                return evaluate(board)
        elif player == 1:
            if np.all(board==0):
                place(board, 1, (1,1))
                player = 2
                if evaluate(board):
                    return evaluate(board)
                continue

            random_place(board, player)
            player = 2
            if evaluate(board):
                return evaluate(board)

def main1():
    random.seed(1)
    board = create_board()
    place(board, 1, (1,1))
    #print(posibilities(board))
    #for _ in range(3):
        #random_place(board, 1)
        #random_place(board, 2)
    #place(board, 2, (1,1))
    #print(row_win(board, 1))
    #print(col_win(board, 1))
    #print(diag_win(board, 2))
    print("player %i has won the game..." %evaluate(board))
    print(board)

def main2():
    random.seed(1)
    result = []
    for _ in range(1000):
        result.append(play_game())
    a = np.array(result)
    print(np.count_nonzero(a==1))

def main3():
    random.seed(1)
    result = []
    for _ in range(1000):
        result.append(play_strategic_game())
    a = np.array(result)
    print(np.count_nonzero(a==1))

if __name__ == "__main__":
    main3()
