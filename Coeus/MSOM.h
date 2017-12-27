#pragma once
#include "SOM.h"

namespace Coeus
{
	class __declspec(dllexport) MSOM : public SOM
	{
	public:
		MSOM(int p_input_dim, int p_dim_x, int p_dim_y, NeuralGroup::ACTIVATION p_activation, double p_alpha, double p_beta);
		~MSOM();

		void activate(Tensor *p_input) override;
		double calc_distance(int p_index) override;

		void reset_context() const;

		NeuralGroup* get_context_group() const { return _context_group; }
		Connection* get_context_lattice() const { return _context_lattice; }

	protected:
		Tensor* calc_distance() override;
		void update_context() const;

		NeuralGroup* _context_group;
		Connection* _context_lattice;

		double _alpha;
		double _beta;
	};
}


