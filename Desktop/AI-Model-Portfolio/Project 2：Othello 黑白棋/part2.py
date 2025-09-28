def countFlipPieces(gameboard, player_color, grid, direction_id):
    if len(grid) != 2:
        return 0
    row_char = grid[0].upper()
    col_char = grid[1].lower()
    row = ord(row_char) - ord('A')
    col = ord(col_char) - ord('a')
    
    if row < 0 or row >= 6 or col < 0 or col >= 6:
        return 0
    
    index = row * 6 + col
    if gameboard[index] != '+':
        return 0
    
    opponent_color = 'X' if player_color == 'O' else 'O'
    
    directions = [
        (-1, 0),   # 0: 上
        (-1, 1),   # 1: 右上
        (0, 1),    # 2: 右
        (1, 1),    # 3: 右下
        (1, 0),    # 4: 下
        (1, -1),   # 5: 左下
        (0, -1),   # 6: 左
        (-1, -1)   # 7: 左上
    ]
    
    if direction_id < 0 or direction_id >= 8:
        return 0
    delta_row, delta_col = directions[direction_id]
    
    current_row = row + delta_row
    current_col = col + delta_col
    count = 0
    
    while True:
        if current_row < 0 or current_row >= 6 or current_col < 0 or current_col >= 6:
            return 0
        current_index = current_row * 6 + current_col
        current_cell = gameboard[current_index]
        
        if current_cell == opponent_color:
            count += 1
        elif current_cell == player_color:
            return count
        else:
            return 0
        current_row += delta_row
        current_col += delta_col

def flipPieces(gameboard, player_color, grid):
    # 將棋盤轉換為列表方便操作
    board_list = list(gameboard)
    
    # 檢查位置是否合法
    if len(grid) != 2:
        return ''.join(board_list)
    row_char = grid[0].upper()
    col_char = grid[1].lower()
    row = ord(row_char) - ord('A')
    col = ord(col_char) - ord('a')
    
    if row < 0 or row >= 6 or col < 0 or col >= 6:
        return ''.join(board_list)
    
    index = row * 6 + col
    if board_list[index] != '+':
        return ''.join(board_list)
    
    # 放置玩家棋子
    board_list[index] = player_color
    
    # 八個方向
    directions = [
        (-1, 0),   # 0: 上
        (-1, 1),   # 1: 右上
        (0, 1),    # 2: 右
        (1, 1),    # 3: 右下
        (1, 0),    # 4: 下
        (1, -1),   # 5: 左下
        (0, -1),   # 6: 左
        (-1, -1)   # 7: 左上
    ]
    
    for dr in range(8):
        delta_row, delta_col = directions[dr]
        num_flips = countFlipPieces(gameboard, player_color, grid, dr)
        
        current_row = row + delta_row
        current_col = col + delta_col
        
        for _ in range(num_flips):
            if 0 <= current_row < 6 and 0 <= current_col < 6:
                idx = current_row * 6 + current_col
                board_list[idx] = player_color
                current_row += delta_row
                current_col += delta_col
            else:
                break
    
    return ''.join(board_list)

# 修正后的输入处理
n = int(input())
for _ in range(n):
    gameboard = input().strip()
    player_num = int(input())
    player_color = 'X' if player_num == 1 else 'O'
    grid = input().strip()
    # 已删除读取direction_id的代码行
    new_board = flipPieces(gameboard, player_color, grid)
    print(new_board)