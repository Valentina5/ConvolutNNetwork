#include "DataSet.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <cstring>

#ifdef _MSC_VER

#include <direct.h>
int chdir(const char *path) { return _chdir(path); }

unsigned long __builtin_bswap32(unsigned long x) {
	return _byteswap_ulong(x);
}

#else

#include <unistd.h>

#endif

DataSet::DataSet(const std::string &work_dir) {
	chdir(work_dir.c_str());

	std::fstream trd("train-images.idx3-ubyte", std::ios::in | std::ios::binary);
	if (!trd)
		throw std::domain_error("train-images.idx3-ubyte could not be opened");

	uint32_t sig;
	uint32_t cnt;

	trd.read((char *)&sig, sizeof(uint32_t));
	trd.read((char *)&cnt, sizeof(uint32_t));
	trd.read((char *)&width, sizeof(uint32_t));
	trd.read((char *)&height, sizeof(uint32_t));

	sig = __builtin_bswap32(sig);
	cnt = __builtin_bswap32(cnt);
	width = __builtin_bswap32(width);
	height = __builtin_bswap32(height);
	assert(sig == 0x0803);

	rawtrain.resize(cnt * width * height);

	trd.read((char *)&rawtrain[0], cnt * width * height);
	trd.close();

	std::fstream trl("train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!trl)
		throw std::domain_error("train-labels.idx1-ubyte could not be opened");

	trl.read((char *)&sig, sizeof(uint32_t));
	trl.read((char *)&cnt, sizeof(uint32_t));

	sig = __builtin_bswap32(sig);
	cnt = __builtin_bswap32(cnt);
	assert(sig == 0x0801);
	
	std::cout << "Train samples are " << width << "x" << height << std::endl;

	assert(cnt * width * height == rawtrain.size());
	labeltrain.resize(cnt);

	trl.read((char *)&labeltrain[0], cnt);
	
	std::cout << "Read " << cnt << " samples in train set" << std::endl;

	std::fstream tsd("t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
	if (!tsd)
		throw std::domain_error("t10k-images.idx3-ubyte could not be opened");
	
	tsd.read((char *)&sig, sizeof(uint32_t));
	tsd.read((char *)&cnt, sizeof(uint32_t));
	tsd.read((char *)&width, sizeof(uint32_t));
	tsd.read((char *)&height, sizeof(uint32_t));

	sig = __builtin_bswap32(sig);
	cnt = __builtin_bswap32(cnt);
	width = __builtin_bswap32(width);
	height = __builtin_bswap32(height);
	assert(sig == 0x0803);
	
	std::cout << "Test samples are " << width << "x" << height << std::endl;

	rawtest.resize(cnt * width * height);

	tsd.read((char *)&rawtest[0], cnt * width * height);
	tsd.close();
	
	std::cout << "Read " << cnt << " samples in test set" << std::endl;

	std::fstream tsl("t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	if (!tsl)
		throw std::domain_error("t10k-labels.idx1-ubyte could not be opened");

	tsl.read((char *)&sig, sizeof(uint32_t));
	tsl.read((char *)&cnt, sizeof(uint32_t));

	sig = __builtin_bswap32(sig);
	cnt = __builtin_bswap32(cnt);
	assert(sig == 0x0801);

	assert(cnt * width * height == rawtest.size());
	labeltest.resize(cnt);

	tsl.read((char *)&labeltest[0], cnt);

	normalize_coeff();
}

void DataSet::normalize_coeff() {

	trainmean = testmean = 34;
	traindev = testdev = 79;

	return;

	trainmean = testmean = traindev = testdev = 0;

	for (int i = 0; i < rawtrain.size(); i++) {
		trainmean += rawtrain[i];
		traindev += rawtrain[i] * rawtrain[i];
	}

	trainmean /= rawtrain.size();
	traindev /= rawtrain.size();
	traindev -= trainmean * trainmean;
	traindev = sqrt(traindev);

	for (int i = 0; i < rawtest.size(); i++) {
		testmean += rawtest[i];
		testdev += rawtest[i] * rawtest[i];
	}
	
	testmean /= rawtest.size();
	testdev /= rawtest.size();
	testdev -= testmean * testmean;
	testdev = sqrt(testdev);

	std::cout << "Train set: mean = " << trainmean << ", dev = " << traindev << std::endl;
	std::cout << "Test set: mean = " << testmean << ", dev = " << testdev << std::endl;
}

int DataSet::getPattern(int i, float X[], bool trainSet) {
	unsigned char *Xraw = (trainSet ? rawtrain.data() : rawtest.data()) + i * 28 * 28;
	float mean = trainSet ? trainmean : testmean;
	float dev = trainSet ? traindev : testdev;

	memset(X, 0, 29 * 29 * sizeof(float));
	for (int i = 0; i < 28; i++)
		for (int j = 0; j < 28; j++)
			X[i * 29 + j] = Xraw[i * 28 + j];

	for (int i = 0; i < 29; i++)
		for (int j = 0; j < 29; j++) {
			X[i * 29 + j] -= mean;
			X[i * 29 + j] /= 3 * dev;
		}
	
	return trainSet ? labeltrain[i] : labeltest[i];
}
