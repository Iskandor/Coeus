#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus
{
	class __declspec(dllexport) SOM : public BaseLayer
	{
	public:
		SOM(string p_id, int p_input_dim, int p_dim_x, int p_dim_y, NeuralGroup::ACTIVATION p_activation);
		SOM(nlohmann::json p_data);
		~SOM();

		void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
		void activate(Tensor* p_input = nullptr) override;
		virtual int find_winner(Tensor* p_input);
		void get_position(int p_index, int& p_x, int& p_y) const;
		int get_position(int p_x, int p_y) const;

		NeuralGroup* get_input_group() const { return _input_group; }
		NeuralGroup* get_lattice() const { return _output_group; }
		Connection*  get_input_lattice() const { return _input_lattice; }

		virtual double calc_distance(int p_index);
		virtual double calc_distance(int p_neuron1, int p_neuron2);

		int dim_x() const { return _dim_x; }
		int dim_y() const { return _dim_y; }

		int get_winner() const { return _winner; }
		void set_input_mask(int* p_mask) { _input_mask = p_mask; }
		void set_conscience(double p_val);
		void init_conscience() const;
		void update_conscience(Tensor* p_input);

		virtual SOM* clone() const;
		void override_params(BaseLayer* p_source) override;

	protected:
		void find_winner(Tensor* p_input, bool p_conscience);
		
		void calc_distance();

		Connection* _input_lattice;

		int _winner;
		int _dim_x;
		int _dim_y;

		Tensor	_dist;
		Tensor	_p;
		Tensor	_bias;
		int*	_input_mask;
		double	_conscience;
		const double B = 1e-4;
	};
}


