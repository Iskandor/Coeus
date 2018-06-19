#include "UBALTest.h"
#include "InputLayer.h"
#include "UBALLayer.h"

using namespace Coeus;

UBALTest::UBALTest()
{
	BaseLayer* p_inputX = _network.add_layer(new InputLayer("inputX", 2));
	BaseLayer* p_hidden = _network.add_layer(new UBALLayer("hidden", 4, NeuralGroup::ACTIVATION::SIGMOID, p_inputX));
	BaseLayer* p_inputY = _network.add_layer(new UBALLayer("inputY", 1, NeuralGroup::ACTIVATION::SIGMOID, p_hidden));
}


UBALTest::~UBALTest()
{
}

void UBALTest::run()
{
}
