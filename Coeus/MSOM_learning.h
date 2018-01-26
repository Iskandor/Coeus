#pragma once

#include <Tensor.h>
#include "MSOM.h"
#include "MSOM_params.h"
#include "Base_SOM_learning.h"

using namespace FLAB;

namespace Coeus
{
	class __declspec(dllexport) MSOM_learning : public Base_SOM_learning
	{
	public:
		explicit MSOM_learning(MSOM *p_msom, MSOM_params *p_params, SOM_analyzer* p_analyzer);
		~MSOM_learning();

		void init_msom(MSOM* p_source);
		void train(Tensor *p_input) override;

	private:
		Tensor	_delta_c;

		MSOM* _msom;
	};
}


