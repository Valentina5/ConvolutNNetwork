#ifndef __NEURAL_NET__
#define __NEURAL_NET__

#include "Layer.h"
#include "Sigmoid.h"
#include <vector>
#include <iostream>

class NeuralNet {
public:
	std::vector<Layer *> layers;
	std::vector<float *> X;

	std::vector<float *> Delta;
	std::vector<float *> dXdw;
	std::vector<float *> dXdb;

	std::vector<float *> dEdw;
	std::vector<float *> dEdb;

	int numOutputs;
public:
	NeuralNet() {
	}

	void appendLayer(Layer *layer) {
		if (layers.empty()) {
			layers.push_back(layer);
			X.push_back(new float[layer->numInputs()]);
			X.push_back(new float[layer->numOutputs()]);
			return;
		}

		if (layers.back()->numOutputs() != layer->numInputs()) {
			throw "Cannot append layer: number of neurons does not match";
		}

		layers.push_back(layer);
		X.push_back(new float[layer->numOutputs()]);
	}

	void finalize() {
		numOutputs = layers.back()->numOutputs();

		std::cout << "Net structure:" << std::endl;

		for (int s = 0; s < layers.size(); s++) {
			Delta.push_back(new float[numOutputs * layers[s]->numOutputs()]);
			dXdw.push_back(new float[numOutputs * layers[s]->numWeights()]);
			dXdb.push_back(new float[numOutputs * layers[s]->numBiases()]);
			dEdw.push_back(new float[layers[s]->numWeights()]);
			dEdb.push_back(new float[layers[s]->numBiases()]);
			std::cout << layers[s]->numInputs() << "neu ==[" << layers[s]->numWeights() << "W + "
				<< layers[s]->numBiases() << "b]=> ";
		}
		std::cout << numOutputs << "neu" << std::endl;
	}

	~NeuralNet() {
		for (int i = 0; i < layers.size(); i++)
			delete layers[i];

		for (int s = 0; s < X.size(); s++)
			delete[] X[s];

		for (int s = 0; s < layers.size(); s++) {
			delete[] Delta[s];
			delete[] dXdw[s];
			delete[] dXdb[s];
			delete[] dEdw[s];
			delete[] dEdb[s];
		}
	}

	float *getInput() {
		return X.front();
	}

	float *getOutput() {
		return X.back();
	}

	const float *getInput() const {
		return X.front();
	}

	const float *getOutput() const {
		return X.back();
	}

	void forward() {
		for (int s = 0; s < layers.size(); s++) {
			layers[s]->forward(X[s], X[s + 1]);
		}
	}

	void backward() {
		int l = layers.size();

		for (int i = 0; i < numOutputs; i++)
			for (int k = 0; k < numOutputs; k++)
				Delta[l - 1][i * numOutputs + k] = (i == k) ? Sigmoid::G(X[l][i]) : 0;

		for (int s = l - 1; s > 0; s--) {
			layers[s]->backward(numOutputs, X[s], Delta[s - 1], Delta[s]);
		}
	}

	void derivatives() {
		for (int s = 0; s < layers.size(); s++) {
			layers[s]->derivativeWeight(numOutputs, X[s], Delta[s], dXdw[s]);
			layers[s]->derivativeBias(numOutputs, Delta[s], dXdb[s]);
		}
	}

	float error(const float T[]) const {
		const float *out = getOutput();

		float err = 0;
		for (int i = 0; i < numOutputs; i++)
			err += (T[i] - out[i]) * (T[i] - out[i]);

		return .5f * err;
	}

	void gradient(const float T[]) {
		const float *out = getOutput();

		for (int s = 0; s < layers.size(); s++) {
			int nW = layers[s]->numWeights();
			int nB = layers[s]->numBiases();
			for (int j = 0; j < nW; j++) {
				float sum = 0;
				for (int i = 0; i < numOutputs; i++)
					sum += (out[i] - T[i]) * dXdw[s][i * nW + j];
				dEdw[s][j] = sum;
			}
			for (int j = 0; j < nB; j++) {
				float sum = 0;
				for (int i = 0; i < numOutputs; i++)
					sum += (out[i] - T[i]) * dXdb[s][i * nB + j];
				dEdb[s][j] = sum;
			}
		}

	}

	void descent(float alpha) {
		for (int s = 0; s < layers.size(); s++) {
			int nW = layers[s]->numWeights();
			int nB = layers[s]->numBiases();

			float *w = layers[s]->getWeights();
			float *b = layers[s]->getBiases();

			for (int j = 0; j < nW; j++)
				w[j] -= alpha * dEdw[s][j];
			for (int j = 0; j < nB; j++)
				b[j] -= alpha * dEdb[s][j];
		}
	}

	float learn(const float T[], float alpha) {
		backward();
		derivatives();
		gradient(T);
		descent(alpha);
		return error(T);
	}
};

#endif
