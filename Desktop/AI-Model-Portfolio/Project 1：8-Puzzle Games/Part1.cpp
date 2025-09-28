#include <iostream>
#include <vector>
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
        if (isSolvable(state, 3)) {
            cout << "YES" << endl;
        } else {
            cout << "NO" << endl;
        }
    }
    system("pause");
}

