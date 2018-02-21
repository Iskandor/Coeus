#pragma once
#include "SOM.h"
#include "SOM_params.h"
#include "Base_SOM_learning.h"

namespace Coeus
{
	class __declspec(dllexport) SOM_learning : public Base_SOM_learning
	{
	public:
		explicit SOM_learning(SOM* p_som, SOM_params* p_params, SOM_analyzer* p_analyzer);
		virtual ~SOM_learning();

		void init_som(SOM* p_source) const;
		void train(Tensor *p_input) override;
		void merge(vector<SOM_learning*> &p_learners);

		SOM* som() const { return _som; }

	private:
		Tensor	_delta_w;
		Tensor	_batch_delta_w;

		SOM* _som;
	};

}


