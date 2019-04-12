#pragma once
#include <string>
#include "Tensor.h"
#include "Coeus.h"

using namespace std;

namespace Coeus
{
	class __declspec(dllexport) Param
	{
	public:
		Param(string p_id, Tensor* p_data);
		Param(const Param&);
		Param& operator=(const Param &);
		~Param();

		string get_id() const { return _id; }
		Tensor* get_data() const { return _data; }

	private:
		string	_id;
		Tensor*	_data;
	};
}