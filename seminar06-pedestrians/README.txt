Для использования в классе World предназначены следующие методы/константы:
    static constexpr int ROAD_LENGTH = 10000; -- длина дороги
    static constexpr int LANES_NUMBER = 4; -- количество полос
    static constexpr int LANE_WIDTH = 30; -- ширина полосы

    Car getMyCar() const -- возвращает нашу машину
    double getRoadWidth() const -- возвращает ширину всей дороги
    void makeStep(double a, double df) -- отдаёт команду автомобилю изменять скорость с ускорением aи повернуть руль на df радиан
    vector<DetectedPedestrian> getPedestrians() const -- возвращает пешеходов
    vector<Obstacle> getObstacles() const -- возвращает препятствия
    vector<Crosswalk> getCrosswalks() const -- возвращает переходы

Для использования в классе Car содержатся константы:
    static constexpr double LENGTH = 40;
    static constexpr double WIDTH = 18;
