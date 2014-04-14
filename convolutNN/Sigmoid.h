#ifndef __SIGMOID_H__
#define __SIGMOID_H__

struct Sigmoid {
	static const float alpha;
	static const float beta;
	static float F(float Y);
	static float G(float X);
};

#endif
