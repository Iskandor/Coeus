#pragma once
#include "BaseLayer.h"

namespace Coeus
{
	class __declspec(dllexport) SOM : public BaseLayer
	{
	public:
		SOM(int p_input_dim, int p_dim_x, int p_dim_y, NeuralGroup::ACTIVATION p_activation);
		~SOM();

		void activate(Tensor *p_input) override;
		virtual int find_winner(Tensor* p_input);
		void get_position(int p_index, int& p_x, int& p_y) const;

		NeuralGroup* get_input_group() { return _groups[_inputGroup]; }
		NeuralGroup* get_lattice() { return _groups[_lattice]; }
		Connection*  get_lattice_connection() { return _connections[_inputGroup + "_" + _lattice]; }

	protected:
		virtual double calc_distance(int p_index);
		virtual Tensor* calc_distance();		

		string _lattice;

		int _winner;
		int _dimX;
		int _dimY;
	};
}


