#pragma once
#include <map>
#include "Tensor.h"
#include "ParamModel.h"

namespace Coeus
{
	class __declspec(dllexport) Gradient
	{
		friend class NetworkGradient;
		friend class NaturalGradient;
	public:
		Gradient();
		explicit Gradient(map<string, Tensor>& p_buffer);
		Gradient(Gradient& p_copy);
		~Gradient();

		void init(ParamModel* p_model);
		void fill(float p_value);
		void fill(Tensor& p_gradient);

		bool is_invalid();

		Tensor& operator[](const string& p_id);
		
		map<string, Tensor>::iterator begin() { return _buffer.begin(); }
		map<string, Tensor>::iterator end() { return  _buffer.end(); }

		Gradient& operator= (const Gradient& p_copy);
		Gradient& operator+= (const Gradient& p_rhs);
		Gradient& operator+= (const map<string, Tensor>& p_rhs);
		Gradient& operator/= (float p_rhs);

		void flatten();
		void reshape();

		Tensor& gradient() { return _gradient; }
		Tensor& hessian() { return _hessian; }

	private:
		int _params_size;
		map<string, Tensor> _buffer;

		Tensor _gradient;
		Tensor _hessian;
	};
}
