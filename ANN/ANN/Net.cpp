#include "stdafx.h"
#include "Net.h"




Net::Net(const std::vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// Now we add neurons to the new layer created
		// <= is used to include a bias neuron to the layer
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);

	}
}

void Net::backProp(const std::vector<double> &targetVals)
{
	// Calculate overall net error (RMS of output neuron errors)
	// This is what the net is trying to minimize

	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:

	m_recentAverageError =
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate gradients on hidden layers

	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::displayLayers()
{
	for (unsigned n = 0; n < m_layers.size()-1; ++n) {
		std::ofstream layerFile;
		layerFile.open("Layer" + std::to_string(n) + "File.txt", std::ofstream::app);
		for (unsigned m = 0; m < m_layers[n].size(); ++m) {
			for (unsigned k = 0; k < m_layers[n][m].getWeight().size(); ++k) {
				layerFile << m_layers[n][m].getWeight()[k].weight << " ";
			}
			layerFile << std::endl;
		}
		layerFile << std::endl;
		layerFile.close();
	}
}

void Net::showLayers()
{
	for (unsigned i = 0; i < m_layers.size(); i++)
	{
		std::cout << "Layer: " << i << std::endl;
		for (unsigned j = 0; j < m_layers[i].size(); j++)
		{
			std::cout << "Neuron: " << j << std::endl;
			for (unsigned k = 0; k < m_layers[i][j].getWeight().size(); k++)
			{
				std::cout << m_layers[i][j].getWeight()[k].weight << std::endl;
			}
		}
	}
}

void Net::feedForward(const std::vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign {latch} the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].feedForward(prevLayer); //prevLayer is an reference to the previous layer
		}
	}

}

void Net::getResults(std::vector<double> &resultVals) const
{
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}
