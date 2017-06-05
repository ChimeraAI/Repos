/*
Description:
	This is a simple fully connected neural net with backpropagation
*/

#include "stdafx.h"
#include <assert.h>
#include <string>
#include <iostream>
#include <fstream>

#include "Neuron.h"
#include "trainingDataProcessing.h"
#include "Net.h"

double Neuron::eta = 0.15; // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]

void showVectorVals(std::string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		std::cout << v[i] << " ";
	}

	std::cout << std::endl;
}

int main()
{
	TrainingDataProcessing trainData("Data.txt");

	// e.g., { 3, 2, 1 }
	std::vector<unsigned> topology;
	trainData.getTopology(topology);

	Net myNet(topology);

	myNet.displayLayers();

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	while (!trainData.isEof()) {
		++trainingPass;
		std::cout << std::endl << "Pass " << trainingPass;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		showVectorVals(": Inputs:", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual output results:
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());
		myNet.backProp(targetVals);

		// I want to see if the weights are actually being updated
		//myNet.showLayers();

		//Write the weights to a text files per pass
		myNet.displayLayers();


		// Report how well the training is working, average over recent samples:
		std::cout << "Net recent average error: "
			<< myNet.getRecentAverageError() << std::endl;
	}

	std::cout << std::endl << "Done" << std::endl;

	system("pause");
    return 0;
}

