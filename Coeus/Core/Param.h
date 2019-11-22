#pragma once
#include <string>
#include "Tensor.h"
#include <map>

using namespace std;

namespace Coeus
{
	class __declspec(dllexport) Param
	{
	public:
		Param(const string& p_id, Tensor* p_data);
		//Param(const Param&);
		//Param& operator=(const Param &);
		~Param();

		string get_id() const { return _id; }
		Tensor* get_data() const { return _data; }

	private:
		string	_id;
		Tensor*	_data;
	};

	class __declspec(dllexport) ParamsContainer
	{
	public:
		ParamsContainer();
		~ParamsContainer();

		string					id;
		map<string, Tensor*>	data;
	};
}