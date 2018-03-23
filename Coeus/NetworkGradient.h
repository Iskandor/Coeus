#pragma once

#include "NeuralNetwork.h"

namespace Coeus {

class __declspec(dllexport) NetworkGradient
{
public:
	NetworkGradient(NeuralNetwork* p_network);
	~NetworkGradient();

private:
	NeuralNetwork*	_network;
};

}