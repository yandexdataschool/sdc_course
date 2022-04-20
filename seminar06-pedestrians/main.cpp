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


double getAcceleration(const Car& /*car*/) {
    return 100;
}

void solve() {
    while (true) {
        mtx.lock();
        world.updatePedestrians();

        [[maybe_unused]] const Car ego = world.getMyCar();

        // comment this line to run simulator in manual mode
        //double a = getAcceleration(ego);
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
