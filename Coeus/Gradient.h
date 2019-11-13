#pragma once
#include <map>
#include "Tensor.h"
#include "ParamModel.h"

namespace Coeus
{
	class __declspec(dllexport) Gradient
	{
	public:
		Gradient();
		explicit Gradient(map<string, Tensor>& p_buffer);
		~Gradient();

		void init(ParamModel* p_model);
		void fill(float p_value);

		Tensor& operator[](const string& p_id);
		
		map<string, Tensor>& get_gradient();

		Gradient& operator += (const Gradient& p_rhs);

	private:
		map<string, Tensor> _buffer;
	};
}
