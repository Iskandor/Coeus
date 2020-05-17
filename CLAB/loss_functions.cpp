#include "loss_functions.h"

/**
 * \brief Calculate Mean-squared error(MSE) loss function value
 * \param p_input prediction
 * \param p_target target
 * \return scalar value of MSE
 */
float mse_function::forward(tensor& p_input, tensor& p_target)
{
	float result = 0;

	float *px = p_input.data();
	float *py = p_target.data();

	for(int i = 0; i < p_input.size(); i++)
	{
		result += pow(*px++ - *py++, 2);
	}

	result *= 1.f / (2.f * p_input.shape(0));

	return result;
}

/**
 * \brief Calculate partial derivative of MSE to the each input value
 * \param p_input prediction
 * \param p_target target
 * \return error tensor with shape(batch_size,prediction_dim)
 */
tensor& mse_function::backward(tensor& p_input, tensor& p_target)
{
	gradient.resize({ p_input.shape(0), p_input.shape(1) });

	float *px = p_input.data();
	float *py = p_target.data();
	float *gx = gradient.data();

	for (int i = 0; i < p_input.size(); i++)
	{
		*gx++ = (*px++ - *py++) / p_input.shape(0);
	}

	return gradient;
}
