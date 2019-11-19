#pragma once
#include "Tensor.h"
#include "IMotivationModule.h"

namespace Coeus
{
	class __declspec(dllexport) CountModule : public IMotivationModule
	{
	public:
		CountModule(int p_state_space_size);
		~CountModule();

		void update(Tensor* p_state);

		float uncertainty_motivation() override;
		float familiarity_motivation() override;
		float intermediate_novelty_motivation(float p_sigma) override;
		float surprise_motivation() override;
		float progress_uncertainty_motivation() override;
		float progress_familiarity_motivation() override;

	private:
		Tensor* _state;
		int* _lookup_table;
	};
}
