#pragma once
#include <random>
#include <vector>
#include "RandomGenerator.h"

using namespace std;

namespace Coeus {

	template <typename T>
	class __declspec(dllexport) ReplayBuffer
	{
		public:
			ReplayBuffer(const int p_size) {
				_size = p_size;
			}

			~ReplayBuffer()
			{
				for (size_t i = 0; i < _buffer.size(); i++) {
					delete _buffer[i];
				}
			}

			void add_item(T* p_item) {
				if (_buffer.size() == _size) {
					delete _buffer[0];
					_buffer.erase(_buffer.begin());
				}
				_buffer.push_back(p_item);
			}

			vector<T*>* get_sample(const size_t p_size) {
				size_t size = p_size;
				_sample.clear();

				if (size > _buffer.size()) size = _buffer.size();

				vector<int> index = RandomGenerator::get_instance().choice(_buffer.size(), size);
				shuffle(index.begin(), index.end(), std::mt19937(std::random_device()()));

				for (int i = 0; i < index.size(); i++) {
					_sample.push_back(_buffer[index[i]]);
				}

				return &_sample;
			}

			size_t get_size() const
			{
				return _buffer.size();
			}

		private:
			vector<T*>	_buffer;
			vector<T*>	_sample;
			size_t _size;


	};
}

