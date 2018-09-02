#pragma once

#include <Tensor.h>
#include "json.hpp"
#include "IActivationFunction.h"
#include "Coeus.h"

using namespace std;
using namespace FLAB;

namespace Coeus {

class __declspec(dllexport) NeuralGroup
{
public:
    NeuralGroup(int p_dim, ACTIVATION p_activation_function, bool p_bias);
	explicit NeuralGroup(nlohmann::json p_data);
    NeuralGroup(NeuralGroup& p_copy);
    ~NeuralGroup(void);

	void integrate(Tensor* p_input, Tensor* p_weights);
    void activate();

    string	get_id() const { return _id; };
    int		get_dim() const { return _dim; };
	bool	is_bias() const { return _bias_flag; };

    void	set_output(Tensor* p_output) const;
    Tensor* get_output() { return &_output; };
	void update_bias(Tensor& p_delta_b);
	void set_bias(Tensor* p_bias) const { _bias.override(p_bias); };
	Tensor* get_bias() { return &_bias; };

    ACTIVATION get_activation_function() const { return activation_function_; };

private:
    string  _id;
    ACTIVATION activation_function_;
	IActivationFunction* _f;

	int     _dim;
    Tensor	_output;
    Tensor	_ap;
	bool	_bias_flag;
	Tensor	_bias;
};

}
