#ifndef __CONVLAYER_H__
#define __CONVLAYER_H__

#include <vector>
#include <cmath>

#include "Sigmoid.h"
#include "Layer.h"

template<int kernDim, int inDim, int outDim>
class ConvLayer : public Layer {
	const int mapsIn, mapsOut;
	const int inSize, outSize, kernSize;
	const int inExitSize, outExitSize;

	float *weights;
	float *bias;
public:
	/*
	Convolve mapsIn inDim x inDim maps to mapsOut outDim x outDim
	*/
	ConvLayer(int mapsIn, int mapsOut) 
		: mapsIn(mapsIn), mapsOut(mapsOut),
		inSize(inDim * inDim), outSize(outDim * outDim), kernSize(kernDim * kernDim),
		inExitSize(inSize * mapsIn), outExitSize(outSize * mapsOut)
	{
		weights = new float[kernDim * kernDim * mapsIn * mapsOut];
		bias = new float[mapsOut];
	}

	int getMapsOut() const {
		return mapsOut;
	}

	int getMapsIn() const {
		return mapsIn;
	}
	
	virtual void randomizeWeights() {
		for (int i = 0; i < numWeights(); i++) {
			weights[i] = 0.1f * sin(100.f * i * i + 1234.5);
		}

		for (int i = 0; i < numBiases(); i++)
			bias[i] = 0.1f * sin(312.f * i + 32.41);
	}

	virtual ~ConvLayer() {
		delete[] weights;
		delete[] bias;
	}

	virtual int numWeights() const {
		return kernDim * kernDim * mapsIn * mapsOut;
	}

	virtual int numBiases() const {
		return mapsOut;
	}

	virtual float *getWeights() {
		return weights;
	}

	virtual float *getBiases() {
		return bias;
	}

	virtual int numInputs() const {
		return mapsIn * inSize;
	}

	virtual int numOutputs() const {
		return mapsOut * outSize;
	}

private:
	void forwardPair(const float in[], float out[], const float w[], const float b, bool first) const {
		for (int i = 0; i < outDim; i++)
			for (int j = 0; j < outDim; j++) {
				int i2 = i * 2;
				int j2 = j * 2;
				float sum = 0;
				for (int ki = 0; ki < kernDim; ki++)
					for (int kj = 0; kj < kernDim; kj++) {
						sum += w[ki * kernDim + kj] * in[(i2 + ki) * inDim + j2 + kj];
					}
				out[i * outDim + j] = (first ? b : out[i * outDim + j]) + sum;
			}
	}

	void nonlinearize(float out[]) const {
		for (int i = 0; i < mapsOut * outSize; i++)
			out[i] = Sigmoid::F(out[i]);
	}
public:
	virtual void forward(const float in[], float out[]) const {
		for (int i = 0; i < mapsOut; i++) {
			for (int j = 0; j < mapsIn; j++) {
				int ij = i * mapsIn + j;
				forwardPair(in + j * inSize, out + i * outSize, weights + ij * kernSize, bias[i], j == 0);
			}
		}

		nonlinearize(out);
	}

private:
	void backwardPair(float deltaIn[], const float deltaOut[], const float w[], bool first) const {
		if (first) {
			for (int i = 0; i < inDim * inDim; i++)
				deltaIn[i] = 0;
		}
		for (int i = 0; i < outDim; i++)
			for (int j = 0; j < outDim; j++) {
				int i2 = 2 * i;
				int j2 = 2 * j;
				for (int ki = 0; ki < kernDim; ki++)
					for (int kj = 0; kj < kernDim; kj++) {
						deltaIn[(i2 + ki) * inDim + j2 + kj] += w[ki * kernDim + kj] * deltaOut[i * outDim + j];
					}
			}
	}

	void multiplyDerivative(const float in[], float deltaIn[]) const {
		for (int i = 0; i < inSize * mapsIn; i++)
			deltaIn[i] *= Sigmoid::G(in[i]);
	}

public:
	virtual void backward(const int nLast, const float in[], float deltaIn[], const float deltaOut[]) const {
		for (int i = 0; i < nLast; i++) {
			for (int k = 0; k < mapsIn; k++)
				for (int j = 0; j < mapsOut; j++) {
					int jk = j * mapsIn + k;
					backwardPair(
						deltaIn + i * inExitSize + k * inSize,
						deltaOut + i * outExitSize + j * outSize,
						weights + jk * kernSize,
						j == 0
					);
				}
			multiplyDerivative(in, deltaIn + i * inExitSize);
		}
	}

	virtual void derivativeWeight(const int nLast, const float in[], const float deltaOut[], float dXdw[]) const {
		for (int i = 0; i < nLast; i++)
			for (int k = 0; k < mapsOut; k++)
				for (int m = 0; m < mapsIn; m++)
					for (int kx = 0; kx < kernDim; kx++)
						for (int ky = 0; ky < kernDim; ky++) {
							float sum = 0;
							for (int xi = 0; xi < outDim; xi++)
								for (int eta = 0; eta < outDim; eta++) {
									int xieta = xi * outDim + eta;
									sum += deltaOut[i * outExitSize + k * outSize + xieta] * 
										in[m * inSize + (2 * xi + kx) * inDim + 2 * eta + ky];
								}
							dXdw[i * mapsIn * mapsOut * kernSize + (k * mapsIn + m) * kernSize + kx * kernDim + ky] = sum;
						}
	}
	
	virtual void derivativeBias(const int nLast, const float deltaOut[], float dXdb[]) const {
		for (int i = 0; i < nLast; i++)
			for (int k = 0; k < mapsOut; k++) {
				float sum = 0;
				for (int xieta = 0; xieta < outSize; xieta++)
					sum += deltaOut[i * outExitSize + k * outSize + xieta];
				dXdb[i * mapsOut + k] = sum;
			}
	}
};

#endif
