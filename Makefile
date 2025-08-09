CXX = g++
CXXFLAGS = -pg -O0 -fno-inline -fopenmp -pthread `pkg-config --cflags opencv4`
LDFLAGS = -pg `pkg-config --libs opencv4`
TARGET = jaaa
SRC = newfile.cpp

all: $(TARGET)
	./$(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET) gmon.out
