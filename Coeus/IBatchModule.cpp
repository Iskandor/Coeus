#include "IBatchModule.h"

using namespace Coeus;

IBatchModule::IBatchModule(const int p_batch): 
	_batch_size(p_batch)
{
}


IBatchModule::~IBatchModule()
= default;
