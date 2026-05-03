CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fopenmp
TARGET   := mpg_solver
SRCS     := game_graph.cpp howard.cpp bellman_ford.cpp solver.cpp main.cpp
OBJS     := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
