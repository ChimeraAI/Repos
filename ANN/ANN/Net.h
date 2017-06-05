#pragma once
#include <vector>
#include "Neuron.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include<string>

class Neuron;

typedef std::vector<Neuron> Layer;

class Net
{
public:
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputVals); // Just reads
	void backProp(const std::vector<double> &targetVals); // Just reads
	void getResults(std::vector<double> &resultVals) const; // It just reads the object and doesn't modify it
	double getRecentAverageError(void) const { return m_recentAverageError; }
	void showLayers();
	void displayLayers();

private:
	std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor = 100; // Number of training samples to average over
};