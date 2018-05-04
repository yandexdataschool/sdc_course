
.PHONY: all clean
.DEFAULT_GOAL=all

CXX = c++
CXXFLAGS := \
    -O2 \
    --std=c++14 \
    -F/usr/local/Cellar/qt/5.10.1/lib \
    -I/usr/local/Cellar/qt/5.10.1/lib/QtWidgets.framework/Headers \
    -I/usr/local/Cellar/qt/5.10.1/lib/QtGui.framework/Headers \
    -I/usr/local/Cellar/qt/5.10.1/lib/QtCore.framework/Headers \
    -framework QtCore \
    -framework QtGui \
    -framework QtWidgets \

main.run: main.cpp visualizer.h Makefile
	$(CXX) $(CXXFLAGS) $< -o $@

all: main.run

logs:
	@mkdir -p $@

clean:
	rm -f *.class *.run *.pyc
