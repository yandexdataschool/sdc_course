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


double getAcceleration(const Car& car) {
  auto crosswalks = world.getCrosswalks();

  std::sort(std::begin(crosswalks), std::end(crosswalks), [](const Crosswalk& lhs, const Crosswalk& rhs) {
    return lhs.lx < rhs.lx;
  });

  const Crosswalk* nearestCrosswalk = nullptr;
  for (const auto& cr : crosswalks) {
    if (cr.lx > car.x + Car::LENGTH) {
      nearestCrosswalk = &cr;
      break;
    }
  }

  if (nearestCrosswalk == nullptr) {
    return 100;
  }

  const double maxDeceleration = 100;
  const double brakingDistance = car.v * car.v / maxDeceleration;

  if (car.x + Car::LENGTH + PedestrianFearRadius < nearestCrosswalk->lx - brakingDistance) {
    return 100;
  }

  const auto peds = world.getPedestrians();
  for (const auto& ped : peds) {
    const double dir_y = sin(ped.yaw) < 0 ? -1 : 1;
    const double dir_x = cos(ped.yaw) < 0 ? -1 : 1;

    const bool is_pedestrian_on_crosswalk = (
      ped.x + PedestrianFearRadius > nearestCrosswalk->lx &&
      ped.x - PedestrianFearRadius < nearestCrosswalk->rx);

    // We introduced new error right here:
    const double distance_to_consider = 100;
    const bool is_moving_to_crosswalk = (
      (ped.x > nearestCrosswalk->lx - distance_to_consider && dir_x > 0) ||
      (ped.x < nearestCrosswalk->rx + distance_to_consider && dir_x < 0));

    const bool is_on_same_lane = (
      ped.y + PedestrianFearRadius > car.y - Car::WIDTH &&
      ped.y - PedestrianFearRadius < car.y + Car::WIDTH);

    const bool is_moving_to_our_lane = (
      (ped.y > car.y && dir_y < 0) || (ped.y < car.y && dir_y > 0));

    if ((is_pedestrian_on_crosswalk || is_moving_to_crosswalk) &&
        (is_on_same_lane || is_moving_to_our_lane)) {
       return -100;
    }
  }
  return 100;
}

void solve() {
    while (true) {
        mtx.lock();
        world.updatePedestrians();

        [[maybe_unused]] const Car ego = world.getMyCar();

        // comment this line to run simulator in manual mode
        double a = getAcceleration(ego);
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
