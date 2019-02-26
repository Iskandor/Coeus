#include "IBatchModule.h"

using namespace Coeus;

IBatchModule::IBatchModule(const int p_batch):
	_batch_size(p_batch)
{
	_gradient_accumulator = new GradientAccumulator(&_gradient);
}


IBatchModule::~IBatchModule()
{
	delete _gradient_accumulator;
}
