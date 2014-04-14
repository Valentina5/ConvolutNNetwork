#ifndef __DATASET_H__
#define __DATASET_H__

#include <vector>
#include <stdint.h>
#include <string>

struct DataSet {
	std::vector<unsigned char> rawtrain;
	std::vector<unsigned char> rawtest;
	std::vector<char> labeltrain;
	std::vector<char> labeltest;

	float trainmean, testmean;
	float traindev, testdev;

	uint32_t width, height;

	DataSet(const std::string &path);
	int getPattern(int i, float X[], bool trainSet = true);
	void normalize_coeff();
};

#endif
