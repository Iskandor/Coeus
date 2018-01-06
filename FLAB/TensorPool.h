#pragma once
#include <map>
#include <stack>

using namespace std;

namespace FLAB {

	class __declspec(dllexport) TensorPool
	{
	public:
		static TensorPool& instance();

		double* get_dbl(unsigned int p_size);
		int*	get_int(unsigned int p_size);
		void	release(double* p_buffer, unsigned p_size);
		void	release(int* p_buffer, unsigned p_size);
	private:
		TensorPool();
		~TensorPool();

		map<unsigned int, stack<double*>*> _pool_dbl;
		map<unsigned int, stack<int*>*> _pool_int;
		int _counter;
	};

}