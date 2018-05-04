#include "visualizer.h"

#include <iostream>
#include <fstream>
#include <tuple>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <chrono>
#include <set>
#include <thread>

using namespace std;

Visualizer v;

const int W = 1000;
const int H = 500;
const double carR = 10;

const double PI = acos(-1.0);
using Key = tuple<int, int, int>;

struct Obstacle {
    double xmin, ymin, xmax, ymax;
};

struct State {
    double f, h;

    double x, y, yaw;

    Key getKey() const {
        return make_tuple(int(x / 4), int(y / 4), int(yaw * 3));
    }
};

vector<Obstacle> obstacles;
State start, finish, currentState;
set<State> states, processedStates;

bool operator<(const State& a, const State& b) {
    const double diff = a.f + a.h - (b.f + b.h);
    if (fabs(diff) > 1e-4) return diff < 0;
    return a.getKey() < b.getKey();
}

double dist(double xa, double ya, double xb, double yb) {
    return sqrt((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb));
}

double dist(double l, double r, double h, double x, double y) {
    if (x < l) return dist(l, h, x, y);
    if (x > r) return dist(r, h, x, y);
    return fabs(h - y);
}

bool intersect(const Obstacle& o, const State& st) {
    return dist(o.xmin, o.xmax, o.ymin, st.x, st.y) < carR ||
           dist(o.xmin, o.xmax, o.ymax, st.x, st.y) < carR ||
           dist(o.ymin, o.ymax, o.xmin, st.y, st.x) < carR ||
           dist(o.ymin, o.ymax, o.xmax, st.y, st.x) < carR;
}

bool goodState(const State& st) {
    for (const auto& o : obstacles)
        if (intersect(o, st))
            return false;

    return true;
}

vector<State> possibleMoves(const State& st, const double step, const double yaw_step) {
    vector<State> res;

    State newState;
    newState.x = st.x + cos(st.yaw) * step;
    newState.y = st.y + sin(st.yaw) * step;
    newState.yaw = st.yaw;

    if (!goodState(newState)) return {};

    res.push_back(newState);
    newState.yaw = fmodf(newState.yaw + yaw_step, 2 * PI);
    res.push_back(newState);
    newState.yaw = fmodf(newState.yaw + 2 * PI - 2 * yaw_step, 2 * PI);
    res.push_back(newState);

    return res;
}

double h(const State& st) {
    return dist(st.x, st.y, finish.x, finish.y);
}

State genState() {
    State st;
    while (true) {
        st.x = rand() % (W - 20) + 10;
        st.y = rand() % (H - 20) + 10;
        if (goodState(st)) return st;
    }
}

void findPath() {
    start = genState();
    finish = genState();

    states.clear();
    processedStates.clear();

    states.insert(start);
    set<Key> seen;

    const double step = 8;
    const double yaw_step = 0.42;

    while (!states.empty()) {
        currentState = *states.begin();
        states.erase(states.begin());
        processedStates.insert(currentState);

        if (dist(currentState.x, currentState.y, finish.x, finish.y) < 5) {
            break;
        }

        for (auto nextState : possibleMoves(currentState, step, yaw_step))
            if (seen.insert(nextState.getKey()).second) {
                 nextState.f = currentState.f + step;
                 nextState.h = h(nextState);
                 states.insert(nextState);
            }

        this_thread::sleep_for(chrono::milliseconds(5));
    }
}

void solve() {
    while (true) {
        findPath();
    }
}

void genMap() {
    size_t n = rand() % 20 + 2;
    obstacles.clear();

    for (size_t i = 0; i < n; i++) {
        while (true) {
            double xmin = rand() % W;
            double xmax = rand() % W;
            if (xmin > xmax) swap(xmin, xmax);

            if (xmax - xmin < 10) continue;

            double ymin = rand() % H;
            double ymax = rand() % H;
            if (ymin > ymax) swap(ymin, ymax);

            if (ymax - ymin < 10) continue;

            if (xmax - xmin < W / 10 || ymax - ymin < H / 10) {
                obstacles.push_back(Obstacle{xmin, ymin, xmax, ymax});
                break;
            }
        }
    }

    assert(obstacles.size() == n);
}

void drawState(const State& st, const double R) {
    const double tx = st.x + cos(st.yaw) * R;
    const double ty = st.y + sin(st.yaw) * R;
    v.p.drawLine(st.x, st.y, tx, ty);
}

int main(int argc, char **argv) {
    cout << "Started solution" << endl;

    v.setSize(W, H);
    genMap();

    // v.setOnMouseClick([&](double x, double y) {
    //     points.emplace_back(x, y);
    // });

    v.setOnKeyPress([&](const QKeyEvent& ev) {
        if (ev.key() == Qt::Key_M) {
            genMap();
        }
    });

    thread solveThread(solve);

    while (true) {
        RenderCycle r(v);

        v.p.setBrush(blackBrush);
        for (const auto& o : obstacles) {
            v.p.drawRect(o.xmin, o.ymin, o.xmax - o.xmin, o.ymax - o.ymin);
        }

        v.p.setPen(blackPen);
        for (const auto& s : processedStates) {
            drawState(s, 2);
        }

        v.p.setPen(redPen);
        for (const auto& s : states) {
            drawState(s, 2);
        }

        v.p.setPen(bluePen);
        drawState(currentState, 2);

        v.p.setPen(greenPen);
        drawState(start, 10);

        v.p.setPen(yellowPen);
        drawState(finish, 10);
    }
}
