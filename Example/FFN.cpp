#include "FFN.h"
#include <ostream>
#include <iostream>
#include "neural_network.h"
#include "sgd.h"
#include "loss_functions.h"
#include <omp.h>
#include "tensor_initializer.h"
#include "adam.h"
#include "radam.h"

using namespace std;

FFN::FFN()
{
}


FFN::~FFN()
{
}

void FFN::run() {

	float input_data[8] = {0,0,0,1,1,0,1,1};
	float target_data[4] = { 0,1,1,0 };

	tensor input({ 4, 2 }, input_data);
	tensor target({ 4, 1 }, target_data);

	neural_network network;

	network.add_layer(new dense_layer("hidden0", 400, activation_function::sigmoid(), tensor_initializer::lecun_uniform(), { 2 }));
	network.add_layer(new dense_layer("hidden1", 300, activation_function::sigmoid(), tensor_initializer::lecun_uniform()));
	network.add_layer(new dense_layer("output", 1, activation_function::sigmoid(), tensor_initializer::lecun_uniform()));
	network.add_connection("hidden0", "hidden1");
	network.add_connection("hidden1", "output");
	network.init();

	cout << network.forward(&input) << endl;

	mse_function loss;
	sgd optimizer(&network, -0.5f, 0.9f, true);
	//radam optimizer(&network, -1e-1);

	const double t0 = omp_get_wtime();
	for (int t = 0; t < 500; t++) {
		tensor& output = network.forward(&input);
		const float error = loss.forward(output, target);
		optimizer.update(loss.backward(output, target));

		cout << "Episode " << t;
		cout << " error: " << error << endl;
	}
	cout << omp_get_wtime() - t0 << " s" << endl;

	cout << network.forward(&input) << endl;
}
