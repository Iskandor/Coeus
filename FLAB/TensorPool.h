#pragma once
#include <map>
#include <stack>

using namespace std;

namespace FLAB {

	class TensorPool
	{
	public:
		static TensorPool* getInstance();

		double* get(unsigned int p_size);
		void	release(double* p_buffer);
	private:
		TensorPool();
		~TensorPool();

		static TensorPool* _instance;

		map<unsigned int, stack<double*>*> _pool;
	};

}

