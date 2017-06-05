// Here is where the training data is made

#include "stdafx.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include "trainingDataProcessing.h"

void TrainingDataProcessing::getTopology(std::vector<unsigned> &topology)
{
	std::string line;
	std::string label;

	getline(m_trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	std::cout << label << std::endl;
	if (this->isEof() || label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

TrainingDataProcessing::TrainingDataProcessing(const std::string filename)
{
	m_trainingDataFile.open(filename);
}

unsigned TrainingDataProcessing::getNextInputs(std::vector<double> &inputVals)
{
	inputVals.clear();

	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingDataProcessing::getTargetOutputs(std::vector<double> &targetOutputVals)
{
	targetOutputVals.clear();

	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}


