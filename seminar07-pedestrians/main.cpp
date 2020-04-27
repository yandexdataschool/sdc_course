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
bool pressed;


double getAcceleration(
    const vector<DetectedPedestrian>& peds,
    const vector<Crosswalk>& crosswalks,
    const Car& car) {

    int closestX = World::ROAD_LENGTH + 1;

    for (const auto& cr : crosswalks)
        if (cr.lx > car.x && cr.lx < closestX)
            closestX = cr.lx;

    const double max_brake = 20;
    const double t = car.v / max_brake;
    const double s = car.v * t + (-max_brake) * t * t / 2;

    if (closestX - 2 * Car::LENGTH > car.x + s) {
        return 100;
    }

    for (const auto& p : peds) {
        if (p.x >= closestX - 60 && p.x <= closestX + 100) {
            if (p.y < car.y - World::LANE_WIDTH && sin(p.yaw) < 0.1)
                continue;
            if (p.y > car.y + World::LANE_WIDTH && sin(p.yaw) > -0.1)
                continue;

            if (car.x + Car::LENGTH / 2 + s * 0.5 > p.x)
                continue;

            return -100;
        }
    }

    return 100;
}

void solve() {
    while (true) {
        mtx.lock();
        world.updatePedestrians();

        const Car ego = world.getMyCar();

        // comment this line to run simulator in manual mode
        double a = getAcceleration(world.getPedestrians(), world.getCrosswalks(), ego);
        // double df = ...

        world.makeStep(a, df);

        if (!pressed) {
            a = 0, df = 0;
        }
        pressed = false;

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
        pressed = true;
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
