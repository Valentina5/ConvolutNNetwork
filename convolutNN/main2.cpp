#include "DataSet.h"
#include "ConvLayer.h"

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

	float I[29*29];
	float J[5][13*13];
	float K[3][5*5];
	float O[10];

	ConvLayer<5, 29, 13> cl1(1, 5);
	ConvLayer<5, 13, 5> cl2(5, 3);
	ConvLayer<5, 5, 1> cl3(3, 10);
	cl1.randomizeWeights();
	cl2.randomizeWeights();
	cl3.randomizeWeights();

	ds.getPattern(1234, I, false);
	cl1.forward(I, (float *)J);
	cl2.forward((float *)J, (float *)K);
	cl3.forward((float *)K, O);

	float D0[10][29*29];
	float D1[10][5*13*13];
	float D2[10][3*5*5];
	float D3[10][10];

	for (int i = 0; i < 10; i++)
		for (int k = 0; k < 10; k++)
			D3[i][k] = (i == k ? 1 : 0) * Sigmoid::G(O[i]);

	cl3.backward(10, (float *)K, (float *)D2, (float *)D3);
	cl2.backward(10, (float *)J, (float *)D1, (float *)D2);
	cl1.backward(10, (float *)I, (float *)D0, (float *)D1);

	float dXdW3[10][3*10*5*5];
	float dXdb3[10][10];

	float dXdW2[10][5*3*5*5];
	float dXdb2[10][3];

	float dXdW1[10][1*5*5*5];
	float dXdb1[10][5];

	cl1.Jac1(10, (float *)I, (float *)D1, (float *)dXdW1);
	cl2.Jac1(10, (float *)J, (float *)D2, (float *)dXdW2);
	cl3.Jac1(10, (float *)K, (float *)D3, (float *)dXdW3);

	cl1.Jac2(10, (float *)D1, (float *)dXdb1);
	cl2.Jac2(10, (float *)D2, (float *)dXdb2);
	cl3.Jac2(10, (float *)D3, (float *)dXdb3);

	float Jalt[5][13*13];
	float Kalt[3][5*5];
	float Oalt[10];

	cl1.weights[20] += 1e-3f;
	cl2.bias[2] += 1e-3f;

	cl1.forward(I, (float *)Jalt);
	cl2.forward((float *)Jalt, (float *)Kalt);
	cl3.forward((float *)Kalt, Oalt);

	for (int i = 0; i < 10; i++) {
		std::cout << Oalt[i] - O[i] << "\t" << (dXdW1[i][20] + dXdb2[i][2]) * 1e-3 << std::endl;
	}

	return 0;
}
