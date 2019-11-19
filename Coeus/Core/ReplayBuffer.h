#pragma once
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
				for (int i = 0; i < _buffer.size(); i++) {
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

			vector<T*>* get_sample(const int p_size) {
				int size = p_size;
				_sample.clear();

				if (size > _buffer.size()) size = _buffer.size();

				vector<int> index = RandomGenerator::get_instance().choice(_buffer.size(), size);
				random_shuffle(index.begin(), index.end());

				for (int i = 0; i < index.size(); i++) {
					_sample.push_back(_buffer[i]);
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
			int _size;


	};
}

