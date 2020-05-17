#include "dense_layer.h"
#include "tensor_initializer.h"


/**
 * \brief Create layer with dense weight matrix
 * \param p_id identifier of the layer
 * \param p_dim layer dimension
 * \param p_activation_function activation function instance, it is freed in the layer destructor 
 * \param p_initializer initializer instance, it is freed in the layer destructor
 * \param p_input_shape shape in the form of initializer list
 */
dense_layer::dense_layer(const std::string p_id, const int p_dim, activation_function* p_activation_function, tensor_initializer* p_initializer, std::initializer_list<int> p_input_shape) :
	_id(p_id),
	_dim(p_dim),
	_weights(nullptr),
	_bias(nullptr),
	_initializer(p_initializer),
	_af(p_activation_function)
{
	_input_dim = get_input_dim(p_input_shape);	
	_op = nullptr;
}

/**
 * \brief Create layer which is exact copy of its source
 * \param p_copy source layer
 */
dense_layer::dense_layer(dense_layer& p_copy)
{
	_id = p_copy._id;
	_dim = p_copy._dim;
	_input_dim = p_copy._input_dim;
	_weights = new param(*p_copy._weights);
	_bias = new param(*p_copy._bias);
	_initializer = new tensor_initializer(*p_copy._initializer);
	_af = activation_function::create(p_copy._af->type());
	_op = nullptr;
}


dense_layer::~dense_layer()
{
	delete _initializer;
	delete _op;
	delete _af;
}

/**
 * \brief Performs calculation of the output based on the input, weights, bias and activation function: o = f(Wx + b)
 * \param p_input input tensor with shape (batch_size,input_dim)
 * \return output tensor with shape (batch_size,dim)
 */
tensor& dense_layer::forward(tensor& p_input)
{
	_output.resize({ p_input.shape(0), _dim });

	_output = _op->forward(p_input);
	_output = _af->forward(_output);

	return _output;
}

/**
 * \brief Performs calculation of partial derivatives of weights and bias based on the input error delta
 * \param p_delta error tensor with shape (batch_size,dim)
 * \return error tensor with shape (batch_size,input_dim)
 */
tensor& dense_layer::backward(tensor& p_delta)
{
	p_delta = _af->backward(p_delta);
	return _op->backward(p_delta);
}

/**
 * \brief Initialization of the layer parameters: weight matrix, bias and linear operator
 * \param p_model parent container (neural network)
 * \param p_input_layers list of all input layers feeding the layer
 */
void dense_layer::init(param_model* p_model, std::vector<dense_layer*>& p_input_layers)
{
	if (_weights == nullptr) {
		int input_dim = _input_dim;

		for(auto layer : p_input_layers)
		{
			input_dim += layer->_dim;
		}
	
		_weights = p_model->add_param({ input_dim, _dim });
		_initializer->init(_weights->params());
	}
	else
	{
		p_model->add_param(_weights);
	}
	if (_bias == nullptr) {
		_bias = p_model->add_param({ _dim });
	}
	else
	{
		p_model->add_param(_bias);
	}

	_op = new linear_operator(_weights, _bias);
}

int dense_layer::get_input_dim(std::initializer_list<int>& p_input_shape) const
{
	int result = 0;

	if (p_input_shape.size() > 0)
	{
		result = 1;
		for (auto shape : p_input_shape)
		{
			result *= shape;
		}
	}

	return result;
}
