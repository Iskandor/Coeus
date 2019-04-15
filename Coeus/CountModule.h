#pragma once
#include "Tensor.h"

namespace Coeus
{
	class __declspec(dllexport) CountModule
	{
	public:
		CountModule(int p_state_space_size);
		~CountModule();

		float get_reward(Tensor* p_state) const;
		void update(Tensor* p_state) const;

	private:
		int* _lookup_table;
	};
}
