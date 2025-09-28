def countColorPieces(gameboard, player_color):
    return gameboard.count('X' if player_color == 'X' else 'O')

def heuristicScore(gameboard, player_color):
    opponent_color = 'O' if player_color == 'X' else 'X'
    return countColorPieces(gameboard, player_color) - countColorPieces(gameboard, opponent_color)

n = int(input())
for _ in range(n):
    line = input().strip()
    # 分离棋盘字符串和玩家编号（根据输入格式可能有黏连情况）
    if line[-1].isdigit():
        gameboard = line[:-1]
        player_num = int(line[-1])
    else:
        gameboard = line
        player_num = int(input().strip())
    
    player_color = 'X' if player_num == 1 else 'O'
    print(countColorPieces(gameboard, player_color))