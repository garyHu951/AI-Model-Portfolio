#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <algorithm>
using namespace std;

const int Ndim = 3;

struct Node {
    string state;
    int g, h;
    string action;
    Node* parent;

    int f() const { return g + h; }
};

struct CompareNode {
    bool operator()(const Node* a, const Node* b) {
        if (a->f() == b->f()) return a->g > b->g;
        return a->f() > b->f();
    }
};

bool isSolvable(const string& state) {
    vector<int> vec;
    for (char c : state) vec.push_back(c - '0');
    int inversions = 0;
    int n = Ndim;
    for (int i = 0; i < vec.size(); i++) {
        for (int j = i + 1; j < vec.size(); j++) {
            if (vec[i] != 0 && vec[j] != 0 && vec[i] > vec[j]) inversions++;
        }
    }
    if (n % 2 == 1) return (inversions % 2 == 0);
    int zeroPos = state.find('0');
    int zeroRow = zeroPos / n;
    return (zeroRow + inversions) % 2 == 0;
}

int manhattanDistance(const string& state) {
    int distance = 0;
    for (int i = 0; i < state.size(); i++) {
        if (state[i] == '0') continue;
        int val = state[i] - '0';
        int targetX = val / Ndim;
        int targetY = val % Ndim;
        int currentX = i / Ndim;
        int currentY = i % Ndim;
        distance += abs(targetX - currentX) + abs(targetY - currentY);
    }
    return distance;
}

vector<pair<string, string>> generateSuccessors(const string& state) {
    vector<pair<string, string>> successors;
    int zeroPos = state.find('0');
    int x = zeroPos / Ndim, y = zeroPos % Ndim;
    vector<pair<int, int>> dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    vector<string> dirNames = {"up", "down", "left", "right"};

    for (int i = 0; i < 4; i++) {
        int nx = x + dirs[i].first;
        int ny = y + dirs[i].second;
        if (nx >= 0 && nx < Ndim && ny >= 0 && ny < Ndim) {
            string newState = state;
            swap(newState[zeroPos], newState[nx * Ndim + ny]);
            successors.emplace_back(dirNames[i], newState);
        }
    }
    return successors;
}

void printPath(Node* node) {
    vector<string> path;
    while (node->parent != nullptr) {
        path.push_back("move 0 to " + node->action);
        node = node->parent;
    }
    reverse(path.begin(), path.end());
    for (const auto& step : path) cout << step << endl;
}

int main() {
    string initialState;
    cin >> initialState;

    if (initialState == "012345678") {
        cout << "It is the goal state." << endl;
        system("pause");
        return 0;
    }

    if (!isSolvable(initialState)) {
        cout << "No solution!!" << endl;
        system("pause");
        return 0;
    }

    priority_queue<Node*, vector<Node*>, CompareNode> pq;
    unordered_map<string, int> visited;

    Node* start = new Node{initialState, 0, manhattanDistance(initialState), "", nullptr};
    pq.push(start);
    visited[initialState] = 0;

    Node* final = nullptr;

    while (!pq.empty()) {
        Node* current = pq.top();
        pq.pop();

        if (current->state == "012345678") {
            final = current;
            break;
        }

        if (current->g > visited[current->state]) continue;

        vector<pair<string, string>> successors = generateSuccessors(current->state);
        for (const auto& succ : successors) {
            string newState = succ.second;
            int new_g = current->g + 1;

            if (visited.find(newState) == visited.end() || new_g < visited[newState]) {
                visited[newState] = new_g;
                Node* newNode = new Node{newState, new_g, manhattanDistance(newState), succ.first, current};
                pq.push(newNode);
            }
        }
    }

    if (final == nullptr) {
        cout << "No solution!!" << endl;
    } else {
        printPath(final);
    }

    system("pause");
    return 0;
}