#include <iostream>
#include <queue>
#include <string>
#include <sstream>

using namespace std;

struct Node {
    int f;
    int insertion_order;
    string state;

    Node(int f_, int order_, const string& state_)
        : f(f_), insertion_order(order_), state(state_)
    {}
};

struct CompareNode {
    bool operator()(const Node& a, const Node& b) {
        if (a.f == b.f) {
            return a.insertion_order > b.insertion_order;
        }
        return a.f > b.f;
    }
};

int main() {
    int N;
    cin >> N;
    cin.ignore(); // Ignore the newline after reading N

    priority_queue<Node, vector<Node>, CompareNode> pq;
    int insertion_order = 0;

    for (int i = 0; i < N; ++i) {
        string action;
        getline(cin, action);

        if (action == "enqueue") {
            string line;
            getline(cin, line);
            istringstream iss(line);
            string state;
            int g, h;
            iss >> state >> g >> h;
            int f = g + h;
            pq.emplace(f, insertion_order, state);
            cout << "Insert OK!" << endl;
            insertion_order++;
        } else if (action == "dequeue") {
            if (pq.empty()) {
                cout << "Queue is empty!!" << endl;
            } else {
                Node top = pq.top();
                pq.pop();
                cout << "Got " << top.state << endl;
            }
        }
    }

    return 0;
}