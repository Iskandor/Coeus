#include "FFN.h"
#include "CNN.h"
#include "RNN.h"
#include "ContinuousTest.h"
#include "MazeExample.h"
#include "MotivationTest.h"
#include <omp.h>
#include "OUNoise.h"

using namespace std;

int main()
{
	/*
	Tensor m({ 2,2 }, Tensor::RANDOM);
	m[0] = 1;
	m[1] = 1;
	m[2] = 0;
	m[3] = 0;
	
	cout << m << endl;
	
	Tensor n = m.pinv();

	cout << n << endl;
	*/
	
	/*
	Tensor in_dim_tensor({ 3 }, Tensor::ZERO);
	in_dim_tensor[0] = 2;
	in_dim_tensor[1] = 5;
	in_dim_tensor[2] = 5;

	int channels = in_dim_tensor[0];
	int extent = 3;
	int padding = 1;
	int stride = 1;
	int batch = 2;

	int h2 = (in_dim_tensor[1] - extent + 2 * padding) / stride + 1;
	int w2 = (in_dim_tensor[2] - extent + 2 * padding) / stride + 1;


	Tensor m({ batch,channels,5,5 }, Tensor::VALUE, 1);
	Tensor o({ batch,channels,7,7 }, Tensor::ZERO);
	Tensor n({ batch * h2 * w2, channels * extent * extent }, Tensor::ZERO);

	Tensor::padding(o, m, padding);
	cout << o << endl;

	Tensor::im2col(&o, &n, extent, padding, stride);
	cout << n << endl;

	o.fill(0);

	Tensor::col2im(&n, &o, extent, padding, stride);
	cout << o << endl;
	*/

	//FFN model;
	//model.run();
	//model.run_ubal();
	//model.run_iris();

	//RNN model;
	//model.run_pack();
	//model.run_pack2();
	//model.test_pack();
	//model.test_pack_cm();
	//model.test_pack_alt();
	//model.run_add_problem();
	//model.run_sin_prediction();
	//model.run_add_problem_gru();

	//CNN model;
	//model.run();
	//model.run_mnist();
	//model.test();
	
	//MazeExample example;
	//example.example_q(15000);
	//example.example_sarsa(30000);
	//example.example_double_q(15000);
	//example.example_deep_q(1000);
	//example.example_actor_critic(1000);
	//example.example_nac(1000);
	//example.example_a2c(1000);
	//example.example_a3c(2000);
	//example.example_ppo(1000);
	
	//example.example_icm(64);
	//example.example_selector(64);

	ContinuousTest test;

	//test.run_simple_cacla(1000);
	//test.run_simple_cacer(1000);
	//test.run_simple_ddpg(1000);
	//test.run_cacla(50000, true);
	//for(int i = 0; i < 3; i++) test.run_ddpg_cart_pole(25000, true);

	vector<int> episodes = {1000, 2500, 5000, 10000, 50000};
	vector<int> hiddens = { 5, 10, 30 };
	int setup_index = 0;


	for (int c = 0; c < 5; c++)
	{
		const float clr = RandomGenerator::get_instance().random(1e-4f, 5e-3f);
		for (int a = 0; a < 5; a++)
		{
			const float alr = RandomGenerator::get_instance().random(1e-4f, 5e-3f);
			for (int e = 0; e < 5; e++)
			{
				const int episode = episodes[e];
				for (int h = 0; h < 3; h++)
				{
					const int hidden = hiddens[h];
					LoggerInstance logger("setup_" + to_string(setup_index) + ".log");
					logger.log("clr=" + to_string(clr));
					logger.log("alr=" + to_string(alr));
					logger.log("episode=" + to_string(episode));
					logger.log("hidden=" + to_string(hidden));
					logger.close();
					for (int i = 0; i < 3; i++) test.run_ddpg_mountain_car(episode, hidden, clr, alr, true);
					setup_index++;
				}
			}
		}
	}
			


	
	//for (int i = 0; i < 6; i++) test.run_ddpg_mountain_car_icm(1000, true);

	/*
	IrisTest iris;

	iris.init();
	iris.run(2000);
	iris.test();
	*/

	//MotivationTest test;

	//test.cart_pole_icm(10000);
	//for (int i = 0; i < 10; i++) test.cart_pole_icm2(2500, true);
	//test.test_icm(1000);
	//test.test_gm2(1000);

	system("pause");

	return 0;
}
