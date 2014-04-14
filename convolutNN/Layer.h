#ifndef __LAYER_H__
#define __LAYER_H__

class Layer {

public:

	virtual ~Layer() { }

	virtual int numInputs() const = 0;
	virtual int numOutputs() const = 0;

	virtual int numWeights() const = 0;
	virtual int numBiases() const = 0;

	virtual float *getWeights() = 0;
	virtual float *getBiases() = 0;

	virtual void randomizeWeights() = 0;

	virtual void forward(const float in[], float out[]) const = 0;
	virtual void backward(const int nLast, const float in[], float deltaIn[], const float deltaOut[]) const = 0;
	virtual void derivativeWeight(const int nLast, const float in[], const float deltaOut[], float dXdw[]) const = 0;
	virtual void derivativeBias(const int nLast, const float deltaOut[], float dXdb[]) const = 0;

};

#endif
