#include "world.h"

#include <iostream>
#include <fstream>
#include <tuple>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>

using namespace std;

World world;
double a, df;

pair<int, double> pedestriansInFront() {
    // 0 - no peds
    // 1 - pedestrians out of road
    // 2 - pedestrians on road
    int havePedestrian = 0;

    int closestCrosswalkX = numeric_limits<int>::max();
    const Car ego = world.getMyCar();
    for (const auto& cr : world.getCrosswalks())
        if (cr.lx > ego.x && cr.lx < closestCrosswalkX)
            closestCrosswalkX = (cr.lx + cr.rx) / 2;

    double distToPed = numeric_limits<double>::max();
    for (const auto& p : world.getPedestrians()) {
        if (p.y < ego.y - Car::WIDTH / 2 && sin(p.yaw) < 0)
            continue;
        if (p.y > ego.y + Car::WIDTH / 2 && sin(p.yaw) > 0)
            continue;
        if (p.x < closestCrosswalkX - 100 || p.x > closestCrosswalkX + 100)
            continue;
        if (p.x > ego.x && p.x < ego.x + 600) {
            if ((p.y > 0 && p.y < world.getRoadWidth())
                || dist(p.x, p.y, closestCrosswalkX, 0) < 64
                || dist(p.x, p.y, closestCrosswalkX, world.getRoadWidth()) < 64) {
                if (havePedestrian == 2) {
                    distToPed = min(dist(ego.x, ego.y, p.x, p.y),
                                    distToPed);
                } else {
                    havePedestrian = 2;
                    distToPed = dist(ego.x, ego.y, p.x, p.y);
                }
            } else {
                havePedestrian = 1;
                distToPed = min(closestCrosswalkX - ego.x + fabs(closestCrosswalkX - p.x),
                                distToPed);
            }
        }
    }

    return {havePedestrian, distToPed};
}

void solve() {
    while (true) {
        world.updatePedestrians();

        double desiredSpeed = 200.0;
        auto checkResult = pedestriansInFront();
        if (checkResult.first == 1) desiredSpeed = min(90.0, checkResult.second / 1.91);
        else if (checkResult.first == 2) desiredSpeed = min(70.0, checkResult.second / 2.88 - 40);

        double a = 0;
        const Car ego = world.getMyCar();
        if (ego.v < desiredSpeed - 2) a = 60;
        if (ego.v > desiredSpeed + 2) a = -60;

        double df = 0;

        world.makeStep(a, df);

        if (world.checkCollisions()) {
            world.init();
        }
        if (world.gameOver()) {
            cerr << world.getTimeSinceStart() << " sec.\n";
            break;
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }
}


int main(int argc, char **argv) {
    srand(1991);
    cout << "Started solution" << endl;

    v.setSize(W, H);

    v.setOnKeyPress([&](const QKeyEvent& ev) {
        if (ev.key() == Qt::Key_W) a = 100;
        if (ev.key() == Qt::Key_S) a = -100;
        if (ev.key() == Qt::Key_A) df = -0.8;
        if (ev.key() == Qt::Key_D) df = 0.8;
    });

    world.init();
    thread solveThread(solve);

    while (true) {
        RenderCycle r(v);

        world.draw();
    }
}
