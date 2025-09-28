#include <iostream>
#include <vector>
#include <string>
using namespace std;

int countInversions(vector<int> &state) {
    int inversions = 0;
    int n = state.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (state[i] > state[j] && state[i] != 0 && state[j] != 0) {
                inversions++;
            }
        }
    }
    return inversions;
}

bool isSolvable(vector<int> &state, int n) {
    int inversions = countInversions(state);
    if (n % 2 == 1) {
        return (inversions % 2 == 0);
    } else {
        int zeroRow = 0;
        for (int i = 0; i < state.size(); i++) {
            if (state[i] == 0) {
                zeroRow = i / n;
                break;
            }
        }
        return ((zeroRow + inversions) % 2 == 0);
    }
}

void generateSuccessors(vector<int> &state, int n) {
    int zeroPos;
    for (int i = 0; i < state.size(); i++) {
        if (state[i] == 0) {
            zeroPos = i;
            break;
        }
    }
    int x0 = zeroPos / n, y0 = zeroPos % n;
    vector<pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    vector<string> moveNames = {"up", "down", "left", "right"};
    vector<pair<string, vector<int>>> successors;
    
    for (int i = 0; i < 4; i++) {
        int newX = x0 + moves[i].first;
        int newY = y0 + moves[i].second;
        if (newX >= 0 && newX < n && newY >= 0 && newY < n) {
            vector<int> newState = state;
            swap(newState[zeroPos], newState[newX * n + newY]);
            successors.push_back({"move 0 to " + moveNames[i], newState});
        }
    }
    
    cout << successors.size() << endl;
    for (auto &s : successors) {
        cout << s.first << endl;
        for (int num : s.second) cout << num;
        cout << endl;
    }
}

int main() {
    int t;
    cin >> t;
    while (t--) {
        string input;
        cin >> input;
        vector<int> state;
        for (char c : input) {
            state.push_back(c - '0');
        }
        generateSuccessors(state, 3);
    }
    system("pause");
    return 0;
}

