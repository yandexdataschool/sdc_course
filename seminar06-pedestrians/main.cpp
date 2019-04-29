#include "world.h"

#include <iostream>
#include <fstream>
#include <tuple>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>

using namespace std;

World world;
double a, df;
mutex mtx;

enum class Status {
    NO_PEDESTRIANS,
    ON_THE_ROAD,
};

pair<Status, double> pedestriansInFront() {
    const Car ego = world.getMyCar();
    Status status = Status::NO_PEDESTRIANS;
    double distance = numeric_limits<double>::max();
    double closest_crosswalk = numeric_limits<double>::max();
    for (const auto& cw : world.getCrosswalks()) {
        if (cw.lx > ego.x - Car::LENGTH / 2. && cw.lx < closest_crosswalk) {
            closest_crosswalk = cw.lx;
        }
    }

    for (const auto& p : world.getPedestrians()) {
      if (p.y + PedestrianFearRadius < ego.y - Car::WIDTH / 2. && sin(p.yaw) < 0) {
         continue;
      }
      if (p.y - PedestrianFearRadius > ego.y + Car::WIDTH / 2. && sin(p.yaw) > 0) {
         continue;
      }
      if (p.x < ego.x - Car::LENGTH / 2. || p.x > closest_crosswalk + 100) {
         continue;
      }
      distance = min(distance, dist(p.x, p.y, ego.x, ego.y));
      status = Status::ON_THE_ROAD;
    }
    return {status, closest_crosswalk - ego.x - Car::LENGTH / 2. - 40};
}

void solve() {
    while (true) {
        mtx.lock();
        world.updatePedestrians();

        double a = 0;
        auto result = pedestriansInFront();
        a = 60;
        const Car ego = world.getMyCar();
        if (result.first == Status::ON_THE_ROAD) {
            if (result.second <= 500) {
                a = min(-ego.v * ego.v / result.second / 2., -1.);
            }
        }

        double df = 0;
        world.makeStep(a, df);

        if (world.checkCollisions()) {
            cerr << "collision in " << world.getTimeSinceStart() << " sec.\n";
            world.init();
        }
        if (world.gameOver()) {
            cerr << world.getTimeSinceStart() << " sec.\n";
            break;
        }
        mtx.unlock();

        this_thread::sleep_for(chrono::milliseconds(10));
    }
}


int main(int /*argc*/, char ** /*argv*/) {
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

    World world_copy;
    while (true) {
        RenderCycle r(v);

        {
            mtx.lock();
            world_copy = world;
            mtx.unlock();
        }
        world_copy.draw();
    }
}
