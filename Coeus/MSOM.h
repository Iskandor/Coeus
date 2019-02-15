#pragma once
#include "SOM.h"

namespace Coeus
{
	class __declspec(dllexport) MSOM : public SOM
	{
	public:
		MSOM(string p_id, int p_input_dim, int p_dim_x, int p_dim_y, ACTIVATION p_activation, float p_alpha, float p_beta);
		explicit MSOM(nlohmann::json p_data);
		~MSOM();

		void activate(Tensor* p_input = nullptr) override;
		float calc_distance(int p_index) override;
		float calc_distance(int p_neuron1, int p_neuron2) override;

		void update_context() const;
		void reset_context() const;

		SimpleCellGroup* get_context_group() const { return _context_group; }
		Connection* get_context_lattice() const { return _context_lattice; }

		float get_alpha() const { return _alpha; }
		float get_beta() const { return _beta; }

		MSOM* clone() const override;
		void override(BaseLayer* p_source) override;

	protected:		

		SimpleCellGroup* _context_group;
		Connection* _context_lattice;

		float _alpha;
		float _beta;
	};
}


