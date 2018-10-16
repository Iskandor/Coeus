#pragma once
#include <map>
#include "Tensor.h"

using namespace FLAB;
using namespace std;

namespace Coeus {
class __declspec(dllexport) ParamModel
{
	public:
		ParamModel();
	virtual ~ParamModel();

	protected:
		Tensor* add_param(const string& p_id, Tensor* p_param);
		void add_param(ParamModel* p_model);

		map<string, Tensor*> _params;
};
}

