#ifndef ANFUNCTORS_H_
#define ANFUNCTORS_H_

#include "../math/Functions.h"


struct saxpy_functor { // Y <- A * X + Y
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
	float operator()(const float& x, const float& y) const {
		return a * x + y;
	}
};

struct sax_functor { // Y <- A * X * Y
    const float a;

    sax_functor(float _a) : a(_a) {}

    __host__ __device__
	float operator()(const float& x) const {
		return a * x;
	}
};

struct saxmy_functor { // Y <- A * (X - Y)
	const float a;

	saxmy_functor(float _a) : a(_a) {}

	__host__ __device__
	float operator()(const float& x, const float& y) const { 
		return a * (x - y);
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
struct bigger_tuple_functor { // return the biggest of two tuples
    __device__ __host__
    thrust::tuple<float, unsigned int> operator() (	
    	const thrust::tuple<float, unsigned int> &a, 
		const thrust::tuple<float, unsigned int> &b ) 
    {
    	return (a >= b) ? a : b;
    }
};

struct smaller_tuple_functor { // return the biggest of two tuples
    __device__ __host__
    thrust::tuple<float, unsigned int> operator() (	
    	const thrust::tuple<float, unsigned int> &a, 
		const thrust::tuple<float, unsigned int> &b ) 
    {
    	return (a <= b) ? a : b;
    }
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
struct minus_pow_functor {
    const float fVal;
    minus_pow_functor(float val) : fVal(val) {}

    __host__ __device__
	float operator()(const float& val) const { 
		return pow(fVal-val, 2);
	}
};

struct sqrt_functor {
    __host__ __device__
	float operator()(const float& val) const { 
		return sqrt(val);
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
struct bubble_functor {
	float fSigmaT;
	bubble_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}

    __host__ __device__
	float operator()(const float& dist) const {
    	return ANN::fcn_bubble_neighborhood(dist, fSigmaT);
	}
};

struct gaussian_functor {
	float fSigmaT;
	gaussian_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}

    __host__ __device__
	float operator()(const float& dist) const {
    	return ANN::fcn_gaussian_bell(dist, fSigmaT);
	}
};

struct cut_gaussian_functor {
	float fSigmaT;
	cut_gaussian_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}

    __host__ __device__
	float operator()(const float& dist) const {
    	return ANN::fcn_cut_gaussian_bell(dist, fSigmaT);
	}
};

struct mexican_functor {
	float fSigmaT;
	mexican_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}

    __host__ __device__
	float operator()(const float& dist) const {
    	return ANN::fcn_mexican_hat(dist, fSigmaT);
	}
};

struct epanechicov_functor {
	float fSigmaT;
	epanechicov_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}

    __host__ __device__
	float operator()(const float& dist) const {
    	return ANN::fcn_epanechicov_neighborhood(dist, fSigmaT);
	}
};

struct hebbian_functor {
	float fLearningRate;
	float fInput;

	hebbian_functor(const float &learning_rate, const float &input) :
		fLearningRate(learning_rate), fInput(input) {}

    __host__ __device__
	float operator()(const float& fWeight, const float& fInfluence) const {
    	return fWeight + (fInfluence*fLearningRate*(fInput-fWeight) );
	}
};

#endif