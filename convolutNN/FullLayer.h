#ifndef __FULLLAYER_H__
#define __FULLLAYER_H__

#include "Layer.h"
#include "Sigmoid.h"

class FullLayer : public Layer {
	const int nIn, nOut;

	float *weights;
	float *bias;
public:
	FullLayer(int nIn, int nOut) : nIn(nIn), nOut(nOut) {
		weights = new float[nIn * nOut];
		bias = new float[nOut];
	}

	virtual ~FullLayer() {
		delete[] weights;
		delete[] bias;
	}

	virtual void randomizeWeights() {
		for (int i = 0; i < nIn * nOut; i++)
			weights[i] = 0.1f * sin(231.52f * i * (3 * i + 1) + 3.15f);
		for (int i = 0; i < nOut; i++)
			bias[i] = 0.1f * sin(312.52f * i * (2 * i + 1.6f) + 13.5f);
	}

	virtual int numInputs() const {
		return nIn;
	}
	virtual int numOutputs() const {
		return nOut;
	}

	virtual int numWeights() const {
		return nIn * nOut;
	}
	virtual int numBiases() const {
		return nOut;
	}

	virtual float *getWeights() {
		return weights;
	}
	virtual float *getBiases() {
		return bias;
	}

	virtual void forward(const float in[], float out[]) const {
		for (int i = 0, inIn = 0; i < nOut; i++, inIn += nIn) {
			float sum = 0;
			for (int j = 0; j < nIn; j++)
				sum += weights[inIn + j] * in[j];
			out[i] = Sigmoid::F(sum);
		}	
	}

	virtual void backward(const int nLast, const float in[], float deltaIn[], const float deltaOut[]) const {
		for (int i = 0; i < nLast; i++)
			for (int k = 0; k < nIn; k++) {
				float sum = 0;
				int inOut = i * nOut;
				for (int j = 0, jnIn = 0; j < nOut; j++, jnIn += nIn)
					sum += deltaOut[inOut + j] * weights[jnIn + k];
				deltaIn[i * nIn + k] = Sigmoid::G(in[k]) * sum;
			}
	}
	virtual void derivativeWeight(const int nLast, const float in[], const float deltaOut[], float dXdw[]) const {
		for (int i = 0, inOut = 0; i < nLast; i++, inOut += nOut) 
			for (int k = 0; k < nOut; k++)
				for (int m = 0; m < nIn; m++)
					dXdw[(inOut + k) * nIn + m] = deltaOut[inOut + k] * in[m];
	}
	virtual void derivativeBias(const int nLast, const float deltaOut[], float dXdb[]) const {
		for (int i = 0; i < nLast; i++)
			for (int k = 0; k < nOut; k++)
				dXdb[i * nOut + k] = deltaOut[i * nOut + k];
	}
};

#endif
