CXX      = g++
CXXFLAGS = -O3 -march=native -std=c++20
LAMBDA   ?= 100

# Multi-threaded by default; use `make SINGLE=1` to disable OpenMP
ifndef SINGLE
CXXFLAGS += -fopenmp
endif

.PHONY: clean all

all: binary/main.cpp binary/*.hpp *.hpp
	$(CXX) $(CXXFLAGS) -DLAMBDA_VAL=$(LAMBDA) -o solver binary/main.cpp

clean:
	rm -f solver
