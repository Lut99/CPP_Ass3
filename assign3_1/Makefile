all: simulation

simulation: simulation.cu timer.cc
	nvcc -o simulation simulation.cu timer.cc

plot: result.txt
	gnuplot plot.gnp
	display plot.png

clean: rm -f *.o simulation