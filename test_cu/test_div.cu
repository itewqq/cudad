#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

__global__ void test_2_para_int(int x, int y, int *output) {
	*output = x / y;
}

int main() {
	using namespace std;
	int x, y;
	// x= 0x7f000000;
	// y= 0xff7fffff;
    // read x and y from stdin
    cin >> x >> y;
	int* d_output;
	cudaMalloc((int**)&d_output, sizeof(int));
	test_2_para_int<<<1, 1>>>(x, y, d_output);
	int output;
	cudaMemcpy(&output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
	cout << "output: " << output << endl;
}