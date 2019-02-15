#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus
{
	class __declspec(dllexport) SOM : public BaseLayer
	{
	public:
		SOM(string p_id, int p_input_dim, int p_dim_x, int p_dim_y, ACTIVATION p_activation);
		SOM(nlohmann::json p_data);
		~SOM();

		SOM* clone() override;

		void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
		void activate(Tensor* p_input = nullptr) override;
		virtual int find_winner(Tensor* p_input);
		void get_position(int p_index, int& p_x, int& p_y) const;
		int get_position(int p_x, int p_y) const;

		SimpleCellGroup* get_lattice() const { return _lattice_group; }
		Connection*  get_afferent() const { return _afferent; }

		virtual float calc_distance(int p_index);
		virtual float calc_distance(int p_neuron1, int p_neuron2);

		int dim_x() const { return _dim_x; }
		int dim_y() const { return _dim_y; }

		int get_winner() const { return _winner; }
		void set_input_mask(int* p_mask) { _input_mask = p_mask; }
		void set_conscience(float p_val);
		void init_conscience() const;
		void update_conscience(Tensor* p_input);

		virtual SOM* clone() const;
		void override(BaseLayer* p_source) override;
		void reset() override {};

	protected:
		void find_winner(Tensor* p_input, bool p_conscience);
		void calc_distance();

	protected:
		SimpleCellGroup* _lattice_group;
		SimpleCellGroup* _input_group;
		Connection*		_afferent;

		int _winner;
		int _dim_x;
		int _dim_y;

		Tensor	_dist;
		Tensor	_p;
		Tensor	_bias;
		int*	_input_mask;
		float	_conscience;
		const float B = 1e-4f;
	};
}


