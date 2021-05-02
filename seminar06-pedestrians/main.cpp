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

double getAcceleration(const Car& ego) {
  Crosswalk nearestCrosswalk{10000, 10000};
  const double maxDeceleration = 50;
  const double brakingDistance = ego.v * ego.v / 2 / maxDeceleration;
  auto crosswalks = world.getCrosswalks();
  std::sort(std::begin(crosswalks), std::end(crosswalks),
            [](const Crosswalk& l, const Crosswalk& r) { return l.lx < r.lx;});
  for (const auto& cr : crosswalks) {
    if (ego.x > cr.rx) {
      continue;
    }
    if (nearestCrosswalk.lx > cr.lx) {
      nearestCrosswalk = cr;
    }

    if (cr.lx - nearestCrosswalk.rx < brakingDistance) {
      nearestCrosswalk.rx = cr.rx;
    }
  }


  for (const auto& ped : world.getPedestrians()) {
    if (ped.y + PedestrianFearRadius < ego.y && ped.yaw < 0) {
      continue;
    }
    if (ped.y - PedestrianFearRadius > ego.y && ped.yaw > 0) {
      continue;
    }
    if (ped.x >= nearestCrosswalk.lx - 40 && ped.x <= nearestCrosswalk.rx + 40) {
      //const double timeToCrosswalk = (nearestCrosswalk.lx - ego.x - PedestrianFearRadius) / std::max(ego.v, 0.001);
      if (nearestCrosswalk.lx - ego.x - 30 - PedestrianFearRadius < brakingDistance) {
          return -100;
      }
    }
  }

  return 100;
}

void solve() {
    while (true) {
        mtx.lock();
        world.updatePedestrians();

        const Car ego = world.getMyCar();
        a = getAcceleration(ego);

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
