#include "Sigmoid.h"

#include <cmath>
	
const float Sigmoid::alpha = 1.7320508075688772935f;
const float Sigmoid::beta = 0.65847894846240835431f;

float Sigmoid::F(float Y) {
	return alpha * tanh(beta * Y);
}

float Sigmoid::G(float X) {
	return beta / alpha * (alpha * alpha - X * X);
}
