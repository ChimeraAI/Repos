#pragma once

#include "stdafx.h"
#include <cmath>
#include <vector>

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &preLayer);
	void calcOutputGradients(double targetVals);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	std::vector<Connection> getWeight();

private:
	static double eta;	// [0.0..1.0] overall net training rate
	static double alpha; // [0.0..n] multiplier of last weight change (momentum)
	static double transferFunction(double x); // Change this to model postsynaptic behavior?
	static double transferFunctionDerivative(double x); // Change this to model postsynaptic behavior?
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	// Remember this is for fully connected net
	std::vector<Connection> m_outputWeights; //Stores change in weights and the weights itself
	unsigned m_myIndex;
	double m_gradient;
};