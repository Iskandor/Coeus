#pragma once

#include <Tensor.h>
#include "json.hpp"
#include "IActivationFunction.h"
#include "Coeus.h"
#include "BaseCellGroup.h"

using namespace std;
using namespace FLAB;

namespace Coeus {

class __declspec(dllexport) SimpleCellGroup : public BaseCellGroup
{
public:
    SimpleCellGroup(int p_dim, ACTIVATION p_activation_function, bool p_bias);
	explicit SimpleCellGroup(nlohmann::json p_data);
    SimpleCellGroup(SimpleCellGroup& p_copy);
	SimpleCellGroup& operator = (const SimpleCellGroup& p_copy);
    ~SimpleCellGroup();

	void integrate(Tensor* p_input, Tensor* p_weights) override;
    void activate() override;

	bool is_bias() const { return _bias_flag; }
	void update_bias(Tensor& p_delta_b);
	void set_bias(Tensor* p_bias) const { _bias.override(p_bias); }
	Tensor* get_bias() { return &_bias; }

	IActivationFunction* get_activation_function() const { return _f; }

private:
	ACTIVATION _activation_function;
	IActivationFunction* _f;

	bool	_bias_flag;
	Tensor	_bias;
};
}
