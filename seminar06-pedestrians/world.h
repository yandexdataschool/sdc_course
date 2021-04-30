#include "visualizer.h"

#include <ctime>
#include <list>

using namespace std;

Visualizer v;

const int W = 1234;
const int H = 700;

const double PI = acos(-1.0);


const double PedestrianFearRadius = 27;


double getTime() {
    return double(clock()) / CLOCKS_PER_SEC;
}

double dist(double xa, double ya, double xb, double yb) {
    return sqrt((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb));
}

double dist(double xa, double ya, double xb, double yb, double xc, double yc) {
    const double ac = dist(xa, ya, xc, yc);
    const double bc = dist(xb, yb, xc, yc);
    const double ab = dist(xa, ya, xb, yb);

    if (ac * ac + ab * ab < bc * bc) return ac;
    if (bc * bc + ab * ab < ac * ac) return bc;

    const double A = yb - ya;
    const double B = xa - xb;
    const double C = -xa * A - ya * B;
    const double T = sqrt(A * A + B * B);

    return fabs(A * xc + B * yc + C) / T;
}

struct Car {
    static constexpr double LENGTH = 40;
    static constexpr double WIDTH = 18;
    double x, y;
    double psi, beta;
    double v;
    double frontAngle;
    double lastUpdateTime;

    void updateAndMove(double a, double dFrontAngle) {
        a = max(-50.0, min(a, 50.0));
        dFrontAngle = max(-0.1, min(dFrontAngle, 0.1));
        double curTime = getTime();
        double elapsed = curTime - lastUpdateTime;

        const double dx = v * cos(psi + beta);
        const double dy = v * sin(psi + beta);
        const double dpsi = v / (Car::LENGTH / 2) * sin(beta);

        frontAngle += dFrontAngle * elapsed;
        if (frontAngle > 0.32) frontAngle = 0.32;
        if (frontAngle < -0.32) frontAngle = -0.32;

        beta = atan(0.5 * tan(frontAngle));
        x += dx * elapsed;
        y += dy * elapsed;
        v += a * elapsed;
        if (v > 200) v = 200;
        if (v < 0) v = 0;
        psi += dpsi * elapsed;

        lastUpdateTime = curTime;
    }

    vector<pair<double, double>> getCorners() const {
        const double dxL = cos(psi) * Car::LENGTH / 2;
        const double dyL = sin(psi) * Car::LENGTH / 2;

        const double dxW = cos(psi + PI / 2) * Car::WIDTH / 2;
        const double dyW = sin(psi + PI / 2) * Car::WIDTH / 2;

        return {
            {x + dxL + dxW, y + dyL + dyW},
            {x + dxL - dxW, y + dyL - dyW},
            {x - dxL - dxW, y - dyL - dyW},
            {x - dxL + dxW, y - dyL + dyW},
            {x + dxL + dxW, y + dyL + dyW}
        };
    }

    void reset(double startX, double startY) {
        x = startX;
        y = startY;
        psi = 0;
        beta = 0;
        v = 100;
        frontAngle = 0;
    }
};

struct DetectedPedestrian {
    double x, y, yaw, speed;
};

struct Pedestrian {
    double x, y, yaw;
    double speed;
    double R;

    list<pair<double, double>> trajectory;

    void move(double elapsed) {
        const double distPassed = elapsed * speed;
        while (!trajectory.empty() && dist(trajectory.front().first, trajectory.front().second, x, y) < R)
            trajectory.erase(trajectory.begin());

        if (trajectory.empty()) return;

        const auto& nextPoint = *trajectory.begin();
        yaw = atan2(nextPoint.second - y, nextPoint.first - x);
        x += distPassed * cos(yaw);
        y += distPassed * sin(yaw);
    }
};

struct Obstacle {
    int x, y, r;
};

struct Crosswalk {
    int lx, rx;
};

class World {
public:
    static constexpr int ROAD_LENGTH = 10000;
    static constexpr int LANES_NUMBER = 4;
    static constexpr int LANE_WIDTH = 30;

    bool gameOver() const {
        return ego.x > ROAD_LENGTH;
    }

    double getTimeSinceStart() const {
        return getTime() - initTime;
    }

    Car getMyCar() const {
        return ego;
    }

    double getRoadWidth() const {
        return LANES_NUMBER * LANE_WIDTH;
    }

    void makeStep(double a, double df) {
        ego.updateAndMove(a, df);
    }

    vector<DetectedPedestrian> getPedestrians() const {
        vector<DetectedPedestrian> peds;
        for (const auto& p : pedestrians)
            peds.push_back(DetectedPedestrian{p.x, p.y, p.yaw, p.speed});
        return peds;
    }

    vector<Obstacle> getObstacles() const {
        return obstacles;
    }

    vector<Crosswalk> getCrosswalks() const {
        return crosswalks;
    }

    void updatePedestrians() {
        const double curTime = getTime();
        const double elapsed = curTime - lastPedUpdateTime;

        for (auto& ped : pedestrians) {
            if (rand() % 10000 == 0) ped.speed = rand() % 30 + 10;
            ped.move(elapsed);
        }

        Car me = ego;
        pedestrians.erase(
            remove_if(pedestrians.begin(), pedestrians.end(),
                      [&me](const Pedestrian& p) { return p.trajectory.empty() || p.x < me.x - 456; }),
            pedestrians.end());

        int magic = curTime * 3;  // :)
        const size_t MAX_PEDESTRIANS_COUNT = 50;
        if (magic != prevMagic && pedestrians.size() < MAX_PEDESTRIANS_COUNT) {
            prevMagic = magic;
            pedestrians.push_back(createPedestrian());
        }

        lastPedUpdateTime = curTime;
    }

    bool checkCollisions() {
        vector<pair<double, double>> corners = ego.getCorners();

        for (const auto& pc : corners) {
            if (pc.second < 0) return true;
            if (pc.second > LANE_WIDTH * LANES_NUMBER) return true;
        }

        for (const auto& o : obstacles)
            if (intersect(corners, o))
                return true;

        for (const auto& p : pedestrians)
            if (intersect(corners, p))
                return true;

        return false;
    }

    void init() {
        genMap();
        pedestrians.clear();
        ego.reset(5 + Car::WIDTH / 2, LANE_WIDTH / 2);
        initTime = getTime();
    }

    void draw() {
        const double offX = max(0.0, ego.x - W / 2);

        v.p.setPen(thickBlackPen);
        v.p.drawLine(0 - offX, H / 2, ROAD_LENGTH - offX, H / 2);
        v.p.drawLine(0 - offX, H / 2 + LANE_WIDTH * LANES_NUMBER,
                     ROAD_LENGTH - offX, H / 2 + LANE_WIDTH * LANES_NUMBER);

        for (int i = 1; i < LANES_NUMBER; i++) {
            for (int s = 0; s < ROAD_LENGTH; s += 80)
                v.p.drawLine(s - offX, H / 2 + i * LANE_WIDTH, s + 40 - offX, H / 2 + i * LANE_WIDTH);
        }

        for (const auto& cr : crosswalks) {
            v.p.setBrush(blackBrush);
            v.p.setPen(blackPen);
            static const int stripeWidth = 8;
            int cy = 0;
            while (cy + stripeWidth <= LANE_WIDTH * LANES_NUMBER) {
                v.p.drawRect(cr.lx - offX, H / 2 + cy, cr.rx - cr.lx, stripeWidth);
                cy += stripeWidth * 2;
            }
            v.p.setBrush(whiteBrush);
            v.p.setPen(whitePen);
            cy = stripeWidth;
            while (cy + stripeWidth <= LANE_WIDTH * LANES_NUMBER) {
                v.p.drawRect(cr.lx - offX, H / 2 + cy, cr.rx - cr.lx, stripeWidth);
                cy += stripeWidth * 2;
            }
        }

        v.p.setPen(thickBlackPen);
        v.p.setBrush(greenBrush);
        for (const auto& o : obstacles)
            v.p.drawEllipse(o.x - o.r - offX, H/2 + o.y - o.r, o.r * 2, o.r * 2);

        v.p.setPen(thickBluePen);
        v.p.setBrush(yellowBrush);
        drawCar(ego, offX);

        v.p.setPen(bluePen);
        for (const auto& p : pedestrians) {
            v.p.setBrush(blueBrush);
            v.p.drawEllipse(p.x - 3 - offX, H/2 + p.y - 3, 7, 7);

            v.p.setBrush(transparentBrush);
            v.p.drawEllipse(p.x - PedestrianFearRadius - offX, H/2 + p.y - PedestrianFearRadius,
                            PedestrianFearRadius * 2, PedestrianFearRadius * 2);
        }

        v.p.setFont(QFont("Verdana", 20));
        auto text = "v = " + QString::number(ego.v)
                    + " | frontAngle = " + QString::number(ego.frontAngle);
        v.p.drawText(W / 2, 50, text);
    }

private:
    Car ego;
    vector<Obstacle> obstacles;
    vector<Crosswalk> crosswalks;
    vector<Pedestrian> pedestrians;
    int prevMagic;
    double lastPedUpdateTime;
    double initTime;

    Pedestrian createPedestrian() {
        Pedestrian p;

        bool goingDown;

        while (true) {
            p.x = rand() % ROAD_LENGTH;
            goingDown = rand() % 2;
            p.y = goingDown ? -10 - rand() % LANE_WIDTH : LANE_WIDTH * LANES_NUMBER + 10 + rand() % LANE_WIDTH / 2;

            const double d = dist(p.x, p.y, ego.x, ego.y);
            if (d < 456 || d > 1991) continue;
            else break;
        }

        const double tx = p.x + rand() % 300 - rand() % 300;
        const double ty = !goingDown ? -10 - rand() % LANE_WIDTH : LANE_WIDTH * LANES_NUMBER + 10 + rand() % LANE_WIDTH / 2;

        double crossingX = 0;
        double bestPathX = numeric_limits<double>::max();

        for (const auto& cr : crosswalks) {
            const double curPathX = fabs(p.x - cr.lx) + fabs(tx - cr.lx);
            if (curPathX < bestPathX - 1 || (curPathX < bestPathX + 1 && rand() % 3 == 0)) {
                bestPathX = curPathX;
                crossingX = cr.lx + 5 + rand() % (cr.rx - cr.lx - 10);
            }
        }

        double curX = p.x;
        double curY = p.y;
        while (fabs(curX - crossingX) > 5) {
            if (curX > crossingX) curX -= 7;
            else curX += 7;
            p.trajectory.emplace_back(curX, curY);
        }

        while (fabs(curY - ty) > 5) {
            if (curY > ty) curY -= 7;
            else curY += 7;
            p.trajectory.emplace_back(curX, curY);
        }

        while (fabs(curX - tx) > 5) {
            if (curX > tx) curX -= 7;
            else curX += 7;
            p.trajectory.emplace_back(curX, curY);
        }

        p.R = rand() % 20 + 5;
        p.speed = rand() % 15 + 20;

        return p;
    }

    bool intersect(const vector<pair<double, double>>& corners, const Obstacle& o) {
        for (int i = 0; i < 4; i++)
            if (dist(corners[i].first, corners[i].second,
                     corners[i+1].first, corners[i+1].second,
                     o.x, o.y) <= o.r)
                return true;

        return false;
    }

    bool intersect(const vector<pair<double, double>>& corners, const Pedestrian& p) {
        const double pdx = cos(p.yaw);
        const double pdy = sin(p.yaw);

        for (int i = 0; i < 4; i++) {
            const double cdx = corners[i].first - p.x;
            const double cdy = corners[i].second - p.y;

            double distFront = cdx * pdx + cdy * pdy;
            if (distFront > 0) distFront /= 2;
            else distFront *= 1.5;

            const double distSide = cdx * pdy - cdy * pdx;

            double radiusToCheck = 3;
            if (p.y > 0 && p.y < LANES_NUMBER * LANE_WIDTH) radiusToCheck = PedestrianFearRadius;
            for (const auto& cr : crosswalks)
                if (p.x >= cr.lx && p.x <= cr.rx)
                    radiusToCheck = PedestrianFearRadius;

            if (sqrt(distFront * distFront + distSide * distSide) < radiusToCheck) {
                return true;
            }
        }

        return false;
    }

    void genMap() {
        crosswalks.clear();
        int cx = W / 4;
        while (true) {
            cx += rand() % 1000 + 100;
            if (cx >= ROAD_LENGTH) break;
            crosswalks.push_back(Crosswalk{cx, cx + 40});
        }

        cx = W / 3;
        obstacles.clear();
        while (true) {
            cx += rand() % 500 + 50;
            if (cx >= ROAD_LENGTH) break;
            while (true) {
                Obstacle ob{cx, rand() % (LANES_NUMBER * LANE_WIDTH), rand() % (LANE_WIDTH / 2) + 5};

                bool failed = false;
                for (const auto& cr : crosswalks)
                    if (ob.x + ob.r > cr.lx && ob.x - ob.r < cr.rx) {
                        failed = true;
                        break;
                    }

                if (!failed) {
                    obstacles.push_back(ob);
                    break;
                } else {
                    cx += rand() % 100 - rand() % 100;
                }
            }
        }

        // obstacles.clear();
    }

    void drawCar(const Car& c, const double offX) {
        vector<pair<double, double>> corners = c.getCorners();

        for (int i = 0; i < 4; i++)
            v.p.drawLine(corners[i].first - offX, corners[i].second + H/2,
                         corners[i+1].first - offX, corners[i+1].second + H/2);
    }
};
