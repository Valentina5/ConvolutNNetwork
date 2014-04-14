#include "DataSet.h"
#include "NeuralNet.h"
#include "ConvLayer.h"
#include "FullLayer.h"

#include <fstream>
#include <iostream>

template<int n>
void flush(float I[n*n], const char *fn) {
	std::fstream f(fn, std::ios::out);
	f << "# vtk DataFile Version 3.0" << std::endl;
	f << "Pattern" << std::endl;
	f << "ASCII" << std::endl;
	f << "DATASET RECTILINEAR_GRID" << std::endl;
	f << "DIMENSIONS " << (n + 1) << " " << (n + 1) << " 1" << std::endl;
	f << "X_COORDINATES " << (n + 1) << " float" << std::endl;
	for (int i = 0; i <= n; i++)
		f << i << " ";
	f << std::endl;
	f << "Y_COORDINATES " << (n + 1) << " float" << std::endl;
	for (int i = 0; i <= n; i++)
		f << i << " ";
	f << std::endl;
	f << "Z_COORDINATES 1 float\n0" << std::endl;
	f << "CELL_DATA " << (n * n) << std::endl;
	f << "SCALARS v float\nLOOKUP_TABLE default" << std::endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			f << I[i][j] << " ";
		f << std::endl;
	}
	f.close();
}

int main() {
	DataSet ds("C:\\Users\\Lenovo\\Documents\\Visual Studio 2010\\Projects\\convolutNN\\Working");

	Layer *l1;
	l1 = new ConvLayer<5, 29, 13>(1, 5);
	Layer *l2;
	l2 = new ConvLayer<5, 13, 5>(5, 10);
	Layer *l3;
	l3 = new FullLayer(l2->numOutputs(), 20);
	Layer *l4;
	l4 = new FullLayer(l3->numOutputs(), 10);

	l1->randomizeWeights();
	l2->randomizeWeights();
	l3->randomizeWeights();
	l4->randomizeWeights();

	l1->randomizeWeights();
	l2->randomizeWeights();
	l3->randomizeWeights();
	l4->randomizeWeights();

	NeuralNet net;

	net.appendLayer(l1);
	net.appendLayer(l2);
	net.appendLayer(l3);
	net.appendLayer(l4);
	net.finalize();

	for (int epoch = 0; epoch < 1000; epoch ++) {
		float err = 0;
		int misses = 0;
		int p;
		for (p = 0; p < 10000; p++) {
			int tr = ds.getPattern(p, net.getInput());
			net.forward();
			const float *out = net.getOutput();
			float T[10];
			int mi = 0;
			for (int i = 0; i < 10; i++) {
				T[i] = (i == tr) ? 1.0f : -1.0f;
				if (out[i] > out[mi])
					mi = i;
			}
			if (mi != tr)
				misses++;
			err += net.learn(T, 0.002f);
		}
		std::cout << "Epoch " << epoch << ", error = " << err << ", misses = " << misses << " (" << 100. * misses / p << " %)";

		misses = misses * 1000 / p;

		int testmiss = 0;
		for (p = 0; p < 10000; p++) {
			int tr = ds.getPattern(p, net.getInput(), false);
			net.forward();
			const float *out = net.getOutput();
			float T[10];
			int mi = 0;
			for (int i = 0; i < 10; i++) {
				T[i] = (i == tr) ? 1.0f : -1.0f;
				if (out[i] > out[mi])
					mi = i;
			}
			if (mi != tr)
				testmiss++;
		}
		std::cout << " Test set misses: " << testmiss << " (" << 100. * testmiss / p << " %)" << std::endl;

		if (misses < 5)
			break;
	}

	return 0;
}
