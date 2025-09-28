#include <bits/stdc++.h>
using namespace std;

/* 
Game Board Layout (6x6):
Columns: a b c d e f
Rows:    A B C D E F
Each cell can be 'X' (Black), 'O' (White), or '+' (Empty)
*/

// First module: Counting flippable pieces in a direction
namespace flip_counter {
    /**
     * Counts how many opponent pieces would be flipped when placing a piece at specified grid
     * in the given direction.
     * 
     * @param board Current game state
     * @param current_player Player making the move (1=Black, 2=White)
     * @param position Grid position in format "Xy" (e.g. "Bb")
     * @param direction Direction index (0-7) clockwise: 0=N, 1=NE, 2=E, etc.
     * @return Number of pieces that would be flipped
     */
    int count_flippable(const vector<vector<int>>& board, int current_player, 
                        const string& position, int direction) {
        // Convert position to coordinates
        int row = position[0] - 'A';
        int col = position[1] - 'a';
        
        // Direction vectors (N, NE, E, SE, S, SW, W, NW)
        const int drow[] = {-1, -1, 0, 1, 1, 1, 0, -1};
        const int dcol[] = {0, 1, 1, 1, 0, -1, -1, -1};
        
        int flips = 0;
        
        // Traverse in the specified direction
        for (int step = 1; step < 6; step++) {
            int new_row = row + drow[direction] * step;
            int new_col = col + dcol[direction] * step;
            
            // Check bounds
            if (new_row < 0 || new_row >= 6 || new_col < 0 || new_col >= 6) 
                return 0;
            
            // Found our own piece - chain is complete
            if (board[new_row][new_col] == current_player) 
                break;
            
            // Found empty cell - chain is broken
            if (board[new_row][new_col] == 0) 
                return 0;
            
            // Count opponent piece
            flips++;
        }
        
        return flips;
    }

    void run() {
        int test_cases;
        cin >> test_cases;
        
        vector<int> results;
        
        while (test_cases--) {
            string board_str, position;
            int player, direction;
            
            cin >> board_str >> player >> position >> direction;
            
            // Parse the board
            vector<vector<int>> board(6, vector<int>(6));
            for (int r = 0; r < 6; r++) {
                for (int c = 0; c < 6; c++) {
                    char cell = board_str[r * 6 + c];
                    board[r][c] = (cell == 'X') ? 1 : (cell == 'O') ? 2 : 0;
                }
            }
            
            results.push_back(count_flippable(board, player, position, direction));
        }
        
        // Output results
        for (int result : results) {
            cout << result << endl;
        }
    }
}

// Second module: State transition after a move
namespace game_state {
    /**
     * Converts string board representation to 2D vector
     */
    vector<vector<int>> parse_board(const string& board_str) {
        vector<vector<int>> board(6, vector<int>(6));
        
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                char cell = board_str[r * 6 + c];
                if (cell == 'X') {
                    board[r][c] = 1;  // Black
                } else if (cell == 'O') {
                    board[r][c] = 2;  // White
                } else {
                    board[r][c] = 0;  // Empty
                }
            }
        }
        
        return board;
    }
    
    /**
     * Converts 2D vector board to string representation
     */
    string serialize_board(const vector<vector<int>>& board) {
        string result;
        result.reserve(36);  // Optimize for 6x6 board
        
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                if (board[r][c] == 1) {
                    result += 'X';
                } else if (board[r][c] == 2) {
                    result += 'O';
                } else {
                    result += '+';
                }
            }
        }
        
        return result;
    }
    
    /**
     * Applies a move to the board and returns the new board state
     */
    string apply_move(const string& board_str, int player, const string& position) {
        auto board = parse_board(board_str);
        int row = position[0] - 'A';
        int col = position[1] - 'a';
        
        // Place the new piece
        board[row][col] = player;
        
        // Direction vectors (N, NE, E, SE, S, SW, W, NW)
        const int dr[] = {-1, -1, 0, 1, 1, 1, 0, -1};
        const int dc[] = {0, 1, 1, 1, 0, -1, -1, -1};
        
        // Check all 8 directions
        for (int dir = 0; dir < 8; dir++) {
            int flip_count = 0;
            
            // Count flippable pieces in this direction
            for (int dist = 1; dist < 6; dist++) {
                int nr = row + dr[dir] * dist;
                int nc = col + dc[dir] * dist;
                
                if (nr < 0 || nr >= 6 || nc < 0 || nc >= 6) {
                    flip_count = 0;
                    break;
                }
                
                if (board[nr][nc] == player) break;
                if (board[nr][nc] == 0) {
                    flip_count = 0;
                    break;
                }
                
                flip_count++;
            }
            
            // Flip pieces if needed
            for (int i = 1; i <= flip_count; i++) {
                int nr = row + dr[dir] * i;
                int nc = col + dc[dir] * i;
                board[nr][nc] = player;
            }
        }
        
        return serialize_board(board);
    }
    
    void run() {
        int test_cases;
        cin >> test_cases;
        
        vector<string> results;
        
        while (test_cases--) {
            string board_str, position;
            int player;
            
            cin >> board_str >> player >> position;
            results.push_back(apply_move(board_str, player, position));
        }
        
        // Output results
        for (const string& result : results) {
            cout << result << endl;
        }
    }
}

// Third module: Count pieces of a specific color
namespace piece_counter {
    /**
     * Counts pieces of a specific color on the board
     * @return Number of pieces of the specified player color
     */
    int count_pieces_of_color(const vector<vector<int>>& board, int player_color) {
        int total = 0;
        
        for (const auto& row : board) {
            for (int cell : row) {
                if (cell == player_color) {
                    total++;
                }
            }
        }
        
        return total;
    }
    
    void run() {
        int test_cases;
        cin >> test_cases;
        
        vector<int> results;
        
        while (test_cases--) {
            string board_str;
            int player;
            
            cin >> board_str >> player;
            
            // Convert to board representation
            vector<vector<int>> board(6, vector<int>(6));
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    char cell = board_str[i * 6 + j];
                    board[i][j] = (cell == 'X') ? 1 : (cell == 'O') ? 2 : 0;
                }
            }
            
            results.push_back(count_pieces_of_color(board, player));
        }
        
        // Output all results
        for (int count : results) {
            cout << count << endl;
        }
    }
}

// Final module: AI player using minimax with alpha-beta pruning
namespace ai_player {
    /*
     * AI player implementation using Minimax algorithm with alpha-beta pruning
     * to find the optimal move for the current player.
     */
    class MinimaxEngine {
    private:
        // Game configuration
        string m_board;
        int m_search_depth;
        int m_player;
        
        // Best move found by the search
        string m_best_move;
        
        // Move evaluations for output
        map<string, int> m_move_scores;
        
        // Direction offsets
        const int ROW_OFFSET[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
        const int COL_OFFSET[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        
        // Search node structure
        struct SearchNode {
            string board;
            int alpha;
            int beta;
            int depth;
            string move;
            int score;
            string best_child_move;
            
            SearchNode() : alpha(INT_MIN), beta(INT_MAX), depth(0), score(INT_MIN) {}
        };
        
    public:
        // Constructor
        MinimaxEngine(const string& board, int depth, int player)
            : m_board(board), m_search_depth(depth), m_player(player) {}
        
        // Get the best move found
        string getBestMove() const {
            return m_best_move;
        }
        
        // Get move scores for debugging/output
        const map<string, int>& getMoveScores() const {
            return m_move_scores;
        }
        
        // Start the minimax search
        void findBestMove() {
            SearchNode root;
            root.board = m_board;
            
            // First gather all possible moves
            auto board_matrix = boardToMatrix(m_board);
            auto valid_moves = generateMoves(board_matrix, m_player);
            
            // Evaluate each move at top level and store scores
            for (const auto& move : valid_moves) {
                auto new_board = makeMove(board_matrix, m_player, move);
                string new_board_str = matrixToBoard(new_board);
                
                SearchNode child;
                child.board = new_board_str;
                child.depth = 1;
                child.move = move;
                
                int score = performMinimaxSearch(child, 3 - m_player, false);
                m_move_scores[move] = score;
            }
            
            // Then perform full search to find best move
            performMinimaxSearch(root, m_player, true);
            m_best_move = root.best_child_move;
        }
        
    private:
        // Board conversion utilities
        vector<vector<int>> boardToMatrix(const string& board_str) const {
            vector<vector<int>> matrix(6, vector<int>(6, 0));
            
            for (int r = 0; r < 6; r++) {
                for (int c = 0; c < 6; c++) {
                    char cell = board_str[r * 6 + c];
                    matrix[r][c] = (cell == 'X') ? 1 : (cell == 'O') ? 2 : 0;
                }
            }
            
            return matrix;
        }
        
        string matrixToBoard(const vector<vector<int>>& matrix) const {
            string board_str;
            board_str.reserve(36);
            
            for (int r = 0; r < 6; r++) {
                for (int c = 0; c < 6; c++) {
                    if (matrix[r][c] == 1) {
                        board_str += 'X';
                    } else if (matrix[r][c] == 2) {
                        board_str += 'O';
                    } else {
                        board_str += '+';
                    }
                }
            }
            
            return board_str;
        }
        
        // Position evaluation function
        int evaluatePosition(const string& board_str, int player) const {
            auto matrix = boardToMatrix(board_str);
            int my_pieces = countPieces(matrix, player);
            int opponent_pieces = countPieces(matrix, 3 - player);
            
            // Simple material advantage heuristic
            return my_pieces - opponent_pieces;
        }
        
        // Count pieces for a player
        int countPieces(const vector<vector<int>>& board, int player) const {
            int count = 0;
            
            for (const auto& row : board) {
                for (int cell : row) {
                    if (cell == player) {
                        count++;
                    }
                }
            }
            
            return count;
        }
        
        // Check if a direction contains flippable pieces
        int checkFlippablePieces(const vector<vector<int>>& board, int player, 
                                const string& position, int direction) const {
            int row = position[0] - 'A';
            int col = position[1] - 'a';
            int flips = 0;
            
            for (int step = 1; step < 6; step++) {
                int nr = row + ROW_OFFSET[direction] * step;
                int nc = col + COL_OFFSET[direction] * step;
                
                if (nr < 0 || nr >= 6 || nc < 0 || nc >= 6) return 0;
                if (board[nr][nc] == player) return flips; 
                if (board[nr][nc] == 0) return 0;
                
                flips++;
            }
            
            return 0;
        }
        
        // Generate all valid moves for a player
        vector<string> generateMoves(const vector<vector<int>>& board, int player) const {
            vector<string> moves;
            
            for (int r = 0; r < 6; r++) {
                for (int c = 0; c < 6; c++) {
                    if (board[r][c] == 0) {  // Empty cell
                        string move = string(1, 'A' + r) + string(1, 'a' + c);
                        
                        // Check if this move would flip any pieces
                        for (int dir = 0; dir < 8; dir++) {
                            if (checkFlippablePieces(board, player, move, dir) > 0) {
                                moves.push_back(move);
                                break;
                            }
                        }
                    }
                }
            }
            
            return moves;
        }
        
        // Apply a move to a board
        vector<vector<int>> makeMove(const vector<vector<int>>& board, int player, const string& move) const {
            auto new_board = board;
            int row = move[0] - 'A';
            int col = move[1] - 'a';
            
            // Place the piece
            new_board[row][col] = player;
            
            // Check all 8 directions for flips
            for (int dir = 0; dir < 8; dir++) {
                int flips = checkFlippablePieces(board, player, move, dir);
                
                // Flip the captured pieces
                for (int i = 1; i <= flips; i++) {
                    int nr = row + ROW_OFFSET[dir] * i;
                    int nc = col + COL_OFFSET[dir] * i;
                    new_board[nr][nc] = player;
                }
            }
            
            return new_board;
        }
        
        // Check if the game is over
        bool isGameOver(const vector<vector<int>>& board) const {
            return generateMoves(board, 1).empty() && generateMoves(board, 2).empty();
        }
        
        // Minimax algorithm with alpha-beta pruning
        int performMinimaxSearch(SearchNode& node, int current_player, bool is_maximizing) {
            // Terminal conditions
            if (node.depth == m_search_depth || isGameOver(boardToMatrix(node.board))) {
                node.score = evaluatePosition(node.board, m_player);
                return node.score;
            }
            
            auto board_matrix = boardToMatrix(node.board);
            auto valid_moves = generateMoves(board_matrix, current_player);
            
            // Pass turn if no moves available
            if (valid_moves.empty()) {
                SearchNode next_node = node;
                next_node.depth++;
                return performMinimaxSearch(next_node, 3 - current_player, !is_maximizing);
            }
            
            int best_score = is_maximizing ? INT_MIN : INT_MAX;
            
            // Evaluate each move
            for (const auto& move : valid_moves) {
                auto new_board = makeMove(board_matrix, current_player, move);
                string new_board_str = matrixToBoard(new_board);
                
                SearchNode child;
                child.board = new_board_str;
                child.alpha = node.alpha;
                child.beta = node.beta;
                child.depth = node.depth + 1;
                child.move = move;
                
                int score = performMinimaxSearch(child, 3 - current_player, !is_maximizing);
                
                if (is_maximizing) {
                    if (score > best_score) {
                        best_score = score;
                        node.best_child_move = move;
                    }
                    node.alpha = max(node.alpha, best_score);
                } else {
                    if (score < best_score) {
                        best_score = score;
                        node.best_child_move = move;
                    }
                    node.beta = min(node.beta, best_score);
                }
                
                // Alpha-beta pruning
                if (node.beta <= node.alpha) {
                    break;
                }
            }
            
            return best_score;
        }
    };
    
    // Main solver function for this module
    void run() {
        int test_cases;
        cin >> test_cases;
        
        vector<string> results;
        
        while (test_cases--) {
            string board_str;
            int player, depth;
            
            cin >> board_str >> player >> depth;
            
            // Create AI engine and find best move
            MinimaxEngine ai(board_str, depth, player);
            ai.findBestMove();
            
            // Display move evaluations - using traditional for loop instead of structured binding
            const map<string, int>& move_scores = ai.getMoveScores();
            for (map<string, int>::const_iterator it = move_scores.begin(); it != move_scores.end(); ++it) {
                cout << it->first << " " << it->second << endl;
            }
            
            string best_move = ai.getBestMove();
            cout << "[" << best_move << "]" << endl;
            results.push_back(best_move);
        }
    }
}

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Uncomment one of these to run the respective module
    // flip_counter::run();
    // game_state::run();
    // piece_counter::run();
    ai_player::run();
    
    return 0;
}