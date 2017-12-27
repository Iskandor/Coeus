#pragma once
#include "BaseLayer.h"
#include "Connection.h"

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

		NeuralGroup* get_input_group() const { return _input_group; }
		NeuralGroup* get_lattice() const { return _output_group; }
		Connection*  get_input_lattice() const { return _input_lattice; }

		virtual double calc_distance(int p_index);

		int dim_x() const { return _dim_x; }
		int dim_y() const { return _dim_y; }

	protected:		
		virtual Tensor* calc_distance();		

		Connection* _input_lattice;

		int _winner;
		int _dim_x;
		int _dim_y;
	};
}


