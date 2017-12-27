#pragma once
#include <map>
#include <stack>

using namespace std;

namespace FLAB {

	class __declspec(dllexport) TensorPool
	{
	public:
		static TensorPool& instance();

		double* get(unsigned int p_size);
		void	release(double* p_buffer, unsigned int p_size);
	private:
		TensorPool();
		~TensorPool();

		map<unsigned int, stack<double*>*> _pool;
		int _counter;
	};

}

