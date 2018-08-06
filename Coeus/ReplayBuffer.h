#pragma once
#include "Tensor.h"
#include <queue>

using namespace FLAB;

namespace Coeus {

	class __declspec(dllexport) ReplayBuffer
	{
		public:
			struct Item
			{
				Tensor s0;
				double a;
				Tensor s1;
				double r;
				bool final;

				Item(Tensor* p_s0, const double p_a, Tensor* p_s1, const double p_r, const bool p_final) {
					s0 = Tensor(*p_s0);
					a = p_a;
					s1 = Tensor(*p_s1);
					r = p_r;
					final = p_final;
				}
			};

			explicit ReplayBuffer(int p_size);
			~ReplayBuffer();

			void add_item(Tensor* p_s0, double p_a, Tensor* p_s1, double p_r, bool p_final);
			vector<Item*>* get_sample(int p_size);
			int get_size() const { return _buffer.size(); }

		private:
			vector<Item*>	_buffer;
			vector<Item*>	_sample;
			int _size;
	};

}

