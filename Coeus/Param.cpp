#include "Param.h"

using namespace Coeus;

Param::Param(const string p_id, Tensor* p_data)
{
	_id = p_id;
	_data = p_data;
}

Param::Param(const Param& p_param)
{
	_id = p_param._id;
	_data = new Tensor(*p_param._data);
}

Param& Param::operator=(const Param& p_param)
{
	_id = p_param._id;
	delete _data;
	_data = new Tensor(*p_param._data);

	return *this;
}

Param::~Param()
{
	delete _data;
}
