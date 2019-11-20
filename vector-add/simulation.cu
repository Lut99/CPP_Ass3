#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "timer.h"
#include <iostream>
#include <fstream>

using namespace std;

/* Write array to file */
void file_write_double_array(const char *filename, double *array, int n) {
    ofstream myfile(filename);
    if (myfile.is_open()) {
        for(int count = 0; count < n; count++){
            myfile << array[count] << "\n";
        }
        myfile.close();
    }
    else {
        cout << "Unable to open file";
    }
}

/* Fill array using sin */
void fill(double *array, int offset, int range, double sample_start,
          double sample_end) {
    int i;
    float dx;

    dx = (sample_end - sample_start) / range;
    for (i = 0; i < range; i++) {
        array[i + offset] = sin(sample_start + i * dx);
    }
}

double *simulate(const int i_max, const int t_max, const int block_size,
    double *old_array, double *current_array, double *next_array){
    
}

int main(int argc, char* argv[]) {
    const int i_max       = atoi(argv[1]);
    const int t_max       = atoi(argv[2]);
    const int block_size = atoi(argv[3]);
    
    timer waveTimer("wave timer");

    double *old_array = calloc(i_max, sizeof(double));
    double *current_array = calloc(i_max, sizeof(double));
    double *next_array = calloc(i_max, sizeof(double));
    
    fill(old_array, 1, imax/4, 0, 2*3.14);
    fill(current_array, 2, imax/4, 0, 2*3.14);

    waveTimer.start();
    result_array = simulate(i_max, t_max, block_size, old_array, current_array, next_array);
    waveTimer.stop();

    cout << waveTimer;
    file_write_double_array("result.txt", result_array, imax);

    free(old_array);
    free(current_array);
    free(next_array);

    return 0;
}
