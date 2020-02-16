#include "ContinuousTest.h"
#include "NeuralNetwork.h"
#include "CoreLayer.h"
#include "TD.h"
#include "CACLA.h"
#include "Encoder.h"
#include "LinearInterpolation.h"
#include "LSTMLayer.h"
#include "DDPG.h"
#include "ICM.h"
#include "RAdam.h"
#include "CACER.h"
#include "Logger.h"
#include "ExponentialInterpolation.h"
#include "MountainCar.h"
#include "ContinuousExploration.h"
#include "ForwardModel.h"

using namespace Coeus;

ContinuousTest::ContinuousTest()
{
}


ContinuousTest::~ContinuousTest()
{
}

void ContinuousTest::run_simple_ddpg(int p_episodes)
{
	int hidden = 4;
	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _environment.STATE_DIM() + _environment.ACTION_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections	
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();
	
	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _environment.STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_actor.add_layer(new CoreLayer("output", _environment.ACTION_DIM(), TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	DDPG agent(&network_critic, RADAM_RULE, 1e-3f, 0.99f, &network_actor, RADAM_RULE, 1e-2f, 10000, 64);


	float reward = 0;
	Tensor action({ _environment.ACTION_DIM() }, Tensor::ZERO);
	Tensor state0;
	Tensor state1;
	float total_reward = 0;
	
	Tensor input({ _environment.STATE_DIM() + _environment.ACTION_DIM() }, Tensor::ZERO);

	const float sigma = 1.f;
	ContinuousExploration exploration;
	exploration.init_gaussian(sigma);
	
	for(int e = 0; e < p_episodes; e++)
	{
		total_reward = 0;
		_environment.reset();
		state0 = _environment.get_state();

		while (!_environment.is_finished())
		{			
			action = exploration.explore(agent.get_action(&state0));

			_environment.do_action(action);			
			total_reward += _environment.get_reward();
			state1 = _environment.get_state();
			reward = _environment.get_reward();

			agent.train(&state0, &action, &state1, reward, _environment.is_finished());

			state0 = state1;
		}
		printf("DDPG SimpleEnv episode %i total reward %0.4f\n", e, total_reward);

		if (e % 100 == 0 && false)
		{
			for (int i = 0; i < 21; i++)
			{
				state0[0] = i * 0.5f;

				network_actor.activate(&state0);
				input.reset_index();
				input.push_back(&state0);
				input.push_back(network_actor.get_output()->at(0));
				network_critic.activate(&input);

				reward = _environment.get_reward(state0);
				printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], (*network_critic.get_output())[0], reward, (*network_actor.get_output())[0]);
			}
			system("pause");
		}
	}

	for (int i = 0; i < 21; i++)
	{
		state0[0] = i * 0.5f;

		network_actor.activate(&state0);
		input.reset_index();
		input.push_back(&state0);
		input.push_back(network_actor.get_output()->at(0));
		network_critic.activate(&input);

		reward = _environment.get_reward(state0);
		printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], (*network_critic.get_output())[0], reward, (*network_actor.get_output())[0]);
	}
	cout << endl;
	
	int step = 0;
	total_reward = 0;
	_environment.reset();
	state0 = _environment.get_state();
	while (!_environment.is_finished())
	{
		step++;
		action = *agent.get_action(&state0);
		_environment.do_action(action);
		reward = _environment.get_reward();
		total_reward += reward;
		state0 = _environment.get_state();
	}
	printf("DDPG SimpleEnv steps %i total reward %0.4f\n", step, total_reward);
}

void ContinuousTest::run_simple_cacla(int p_episodes)
{
	int hidden = 32;
	
	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _environment.STATE_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections	
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _environment.STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_actor.add_layer(new CoreLayer("output", _environment.ACTION_DIM(), TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	CACLA actor(&network_critic, RADAM_RULE, 1e-3f, 0.99f, &network_actor, RADAM_RULE, 1e-3f);

	float reward = 0;
	float delta = 0;
	Tensor action({ _environment.ACTION_DIM() }, Tensor::ZERO);
	Tensor state0;
	Tensor state1;
	float total_reward = 0;

	const float sigma = 1.f;
	ContinuousExploration exploration;
	exploration.init_gaussian(sigma);

	for (int e = 0; e < p_episodes; e++)
	{
		total_reward = 0;
		_environment.reset();
		state0 = _environment.get_state();

		while (!_environment.is_finished())
		{
			action = exploration.explore(actor.get_action(&state0));

			_environment.do_action(action);
			reward = _environment.get_reward();
			total_reward += reward;
			state1 = _environment.get_state();

			//cout << _environment.get_state() << " " << reward << endl;
			actor.train(&state0, &action, &state1, reward, _environment.is_finished());

			state0 = state1;
		}
		printf("CACLA SimpleEnv episode %i total reward %0.4f\n", e, total_reward);
	}

	for (int i = 0; i < 21; i++)
	{
		state0[0] = i * 0.5f;

		network_actor.activate(&state0);
		network_critic.activate(&state0);

		reward = _environment.get_reward(state0);
		printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], (*network_critic.get_output())[0], reward, (*network_actor.get_output())[0]);
	}

	int step = 0;
	total_reward = 0;
	_environment.reset();
	state0 = _environment.get_state();
	while (!_environment.is_finished())
	{
		step++;
		action = *actor.get_action(&state0);
		_environment.do_action(action);
		reward = _environment.get_reward();
		total_reward += reward;
		state0 = _environment.get_state();
	}
	printf("CACLA SimpleEnv steps %i total reward %0.4f\n", step, total_reward);
}

void ContinuousTest::run_simple_cacer(int p_episodes)
{
	int hidden = 32;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _environment.STATE_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections	
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _environment.STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_actor.add_layer(new CoreLayer("output", _environment.ACTION_DIM(), TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	CACER agent(&network_critic, RADAM_RULE, 1e-3f, 0.99f, &network_actor, RADAM_RULE, 1e-3f, 10000, 64);


	float reward = 0;
	float delta = 0;
	Tensor action({ _environment.ACTION_DIM() }, Tensor::ZERO);
	Tensor state0;
	Tensor state1;
	float total_reward = 0;
	
	const float sigma = 1.f;
	ContinuousExploration exploration;
	exploration.init_gaussian(sigma);	

	for (int e = 0; e < p_episodes; e++)
	{
		total_reward = 0;
		_environment.reset();
		state0 = _environment.get_state();

		while (!_environment.is_finished())
		{
			action = exploration.explore(agent.get_action(&state0));

			_environment.do_action(action);
			reward = _environment.get_reward();
			total_reward += reward;
			state1 = _environment.get_state();

			//cout << _environment.get_state() << " " << reward << endl;
			agent.train(&state0, &action, &state1, reward, _environment.is_finished());

			state0 = state1;
		}
		printf("CACER SimpleEnv episode %i total reward %0.4f\n", e, total_reward);
		
		if (e % 100 == 0 && false)
		{
			for (int i = 0; i < 21; i++)
			{
				state0[0] = i * 0.5f;

				network_actor.activate(&state0);
				network_critic.activate(&state0);

				reward = _environment.get_reward(state0);
				printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], (*network_critic.get_output())[0], reward, (*network_actor.get_output())[0]);
			}
			system("pause");
		}
	}

	for (int i = 0; i < 21; i++)
	{
		state0[0] = i * 0.5f;

		network_actor.activate(&state0);
		network_critic.activate(&state0);

		reward = _environment.get_reward(state0);
		printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], (*network_critic.get_output())[0], reward, (*network_actor.get_output())[0]);
	}

	int step = 0;
	total_reward = 0;
	_environment.reset();
	state0 = _environment.get_state();
	while (!_environment.is_finished())
	{
		step++;
		action = *agent.get_action(&state0);
		_environment.do_action(action);
		reward = _environment.get_reward();
		total_reward += reward;
		state0 = _environment.get_state();
	}
	printf("CACER SimpleEnv steps %i total reward %0.4f\n", step, total_reward);
}

void ContinuousTest::run_cacla_cart_pole(const int p_episodes, bool p_log)
{
	const int hidden = 512;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), CartPole::STATE));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), CartPole::STATE));
	network_actor.add_layer(new CoreLayer("output", CartPole::ACTION, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "output");
	network_actor.init();

	CACLA actor(&network_critic, RADAM_RULE, 1e-3f, 0.99, &network_actor, RADAM_RULE, 1e-4f);
	//CACER actor(&network_critic, RADAM_RULE, 1e-3f, 0.99, &network_actor, RADAM_RULE, 1e-4f, 10000, 64);
	
	Tensor action({ CartPole::ACTION }, Tensor::ZERO);
	Tensor state0({ CartPole::STATE }, Tensor::ZERO);
	Tensor state1({ CartPole::STATE }, Tensor::ZERO);

	LoggerInstance logger;

	if (p_log) logger = Logger::instance().init();

	const float sigma = 1.f;
	ContinuousExploration exploration;
	exploration.init_gaussian(sigma);

	for (int e = 0; e < p_episodes; ++e) {
		//printf("CartPole episode %i...\n", e);
		_cart_pole.reset();

		copy_state(_cart_pole.get_state(true), state0);

		float total_reward = test_cart_pole(network_actor, network_critic, 195);
		int total_steps = 0;		

		while (true) {
			action = exploration.explore(actor.get_action(&state0));

			_cart_pole.perform_action(action[0]);
			//cout << network_actor.get_output()->at(0) << " " << action[0] << " " << _cart_pole.to_string();
			//cout << state0 << endl;
			copy_state(_cart_pole.get_state(true), state1);

			total_steps += 1;

			actor.train(&state0, &action, &state1, _cart_pole.get_reward(), _cart_pole.is_finished());

			state0 = state1;

			if (e % 1000 == 0) {
				//cout << network_critic.get_output()->at(0) << " " << delta << endl;
			}

			if (_cart_pole.is_finished()) {				
				break;
			}
		}

		cout << "CACLA Episode " << e << " ";

		const float avg_reward = evaluate_cart_pole(total_reward);

		if (avg_reward >= 195)
		{
			//break;
		}
		if (p_log) logger.log(to_string(avg_reward));
		/*
		if (e % 10 == 0) {
			if (test_cart_pole(network_actor, network_critic, 6000) == 6000) {
				break;
			}
		}
		*/

		//printf("CartPole episode %i finished in %i steps with reward %0.2f\n", e, total_steps, total_reward);
		//printf("%i / %i\n", win, lost);
		
	}

	if (p_log) logger.close();
}

void ContinuousTest::run_ddpg_cart_pole(int p_episodes, bool p_log)
{
	_rewards.clear();
	const int hidden = 24;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), CartPole::STATE + CartPole::ACTION));
	network_critic.add_layer(new CoreLayer("hidden1", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), CartPole::STATE));
	network_actor.add_layer(new CoreLayer("hidden1", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_actor.add_layer(new CoreLayer("output", CartPole::ACTION, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	DDPG agent(&network_critic, RADAM_RULE, 1e-4f, 0.99f, &network_actor, RADAM_RULE, 2e-4f, 10000, 64);

	Tensor action({ CartPole::ACTION }, Tensor::ZERO);
	Tensor state0({ CartPole::STATE }, Tensor::ZERO);
	Tensor state1({ CartPole::STATE }, Tensor::ZERO);

	LoggerInstance logger;

	if (p_log) logger = Logger::instance().init();

	float sigma = 1.f;
	ContinuousExploration exploration;
	exploration.init_gaussian(sigma);
	
	
	for (int e = 0; e < p_episodes; ++e) {
		//printf("CartPole episode %i...\n", e);
		const float total_reward = test_cart_pole(network_actor, network_critic, 195);
		int total_steps = 0;

		_cart_pole.reset();
		copy_state(_cart_pole.get_state(true), state0);
		
		while (true) {
			action = exploration.explore(agent.get_action(&state0));

			_cart_pole.perform_action(action[0]);
			//network_critic.activate(&state0);
			//cout << network_critic.get_output()->at(0) << " " << network_actor.get_output()->at(0) << " " << action[0] << " " << _cart_pole.to_string() << endl;
			//cout << state0 << endl;
			copy_state(_cart_pole.get_state(true), state1);

			//total_reward += _cart_pole.get_reward();
			total_steps += 1;

			agent.train(&state0, &action, &state1, _cart_pole.get_reward(), _cart_pole.is_finished());

			state0 = state1;

			if (e % 1000 == 0) {
				//cout << network_critic.get_output()->at(0) << " " << delta << endl;
			}

			if (_cart_pole.is_finished()) {
				break;
			}
		}

		

		//sigma = interpolation.interpolate(e);

		cout << "DDPG Episode " << e << " steps " << total_steps << " ";

		const float avg_reward = evaluate_cart_pole(total_reward);
		
		if (avg_reward >= 195)
		{
			//break;
		}

		if (p_log) logger.log(to_string(avg_reward));

		/*
		if (e % 10 == 0) {
			cout << e << " : ";
			if (test_cart_pole(network_actor, network_critic, 6000) == 6000) {
				break;
			}
		}
		*/

		//printf("CartPole episode %i finished in %i steps with reward %0.2f\n", e, total_steps, total_reward);
		//printf("%i / %i\n", win, lost);

	}

	if (p_log) logger.close();

	test_cart_pole(network_actor, network_critic, 195);
}

void ContinuousTest::run_ddpg_mountain_car(const string& p_dir, int p_episodes, int p_hidden, float clr, float alr, bool p_log)
{
	const int hidden = p_hidden;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _mountain_car.STATE_DIM() + _mountain_car.ACTION_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_critic.add_layer(new CoreLayer("output", 1, LINEAR, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _mountain_car.STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_actor.add_layer(new CoreLayer("output", _mountain_car.ACTION_DIM(), TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	DDPG agent(&network_critic, ADAM_RULE, clr, 0.99f, &network_actor, ADAM_RULE, alr, 100000, 64);

	Tensor action;
	Tensor state0;
	Tensor state1;

	LoggerInstance logger;
	LoggerInstance test_logger;
	if (p_log) {
		logger.init(p_dir);
	}
	//if (p_log) test_logger = Logger::instance().init();

	ContinuousExploration exploration;
	exploration.init_ounoise(_mountain_car.ACTION_DIM(), 0.4f);
	//exploration.init_gaussian(0.2f);

	for (int e = 0; e < p_episodes; ++e) {
		int total_steps = 0;

		_mountain_car.reset();
		state0  = _mountain_car.get_state();

		while (!_mountain_car.is_finished()) {
			total_steps++;
			action = exploration.explore(agent.get_action(&state0));

			_mountain_car.do_action(action);
			state1 = _mountain_car.get_state();
			agent.train(&state0, &action, &state1, _mountain_car.get_reward(), _mountain_car.is_finished());

			state0 = state1;
		}
		exploration.reset();

		//test_logger.log(to_string(e));
		cout << "DDPG Mountain car Episode " << e << " ";
		float reward = test_mountain_car(network_actor);
		
		if (p_log) logger.log(to_string(reward));		
	}

	//if (p_log) test_mountain_car(network_actor, &test_logger);
	
	if (p_log) logger.close();
	//if (p_log) test_logger.close();
}

void ContinuousTest::run_ddpg_mountain_car_icm(int p_episodes, bool p_log)
{
	const int hidden = 10;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _mountain_car.STATE_DIM() + _mountain_car.ACTION_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_critic.add_layer(new CoreLayer("output", 1, LINEAR, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _mountain_car.STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_actor.add_layer(new CoreLayer("output", _mountain_car.ACTION_DIM(), TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	DDPG agent(&network_critic, ADAM_RULE, 1e-3f, 0.99f, &network_actor, ADAM_RULE, 1e-4f, 100000, 64);

	NeuralNetwork network_forward_model;
	network_forward_model.add_layer(new CoreLayer("fm_input", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _mountain_car.STATE_DIM() + _mountain_car.ACTION_DIM()));
	network_forward_model.add_layer(new CoreLayer("fm_output", _mountain_car.STATE_DIM(), TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_forward_model.add_connection("fm_input", "fm_output");
	network_forward_model.init();

	ForwardModel forward_model(&network_forward_model, ADAM_RULE, 1e-4f);
	
	Tensor action;
	Tensor state0;
	Tensor state1;

	LoggerInstance logger;
	LoggerInstance test_logger;
	if (p_log) logger = Logger::instance().init();

	ContinuousExploration exploration;
	exploration.init_ounoise(_mountain_car.ACTION_DIM(), 0.4f);
	//exploration.init_gaussian(0.2f);

	for (int e = 0; e < p_episodes; ++e) {
		int total_steps = 0;
		float cri = 0;
		float cre = 0;

		_mountain_car.reset();
		state0 = _mountain_car.get_state();

		while (!_mountain_car.is_finished()) {
			total_steps++;
			action = exploration.explore(agent.get_action(&state0));

			_mountain_car.do_action(action);
			state1 = _mountain_car.get_state();

			const float ri = forward_model.train(&state0, &action, &state1);
			cri += ri;
			const float re = _mountain_car.get_reward();
			cre += re;

			agent.train(&state0, &action, &state1, re + ri, _mountain_car.is_finished());
			state0 = state1;
		}
		exploration.reset();

		cout << "DDPG Mountain car Episode " << e << " ri " << cri << " ";
		float reward = test_mountain_car(network_actor);

		if (p_log) logger.log(to_string(reward));
	}

	if (p_log) logger.close();	
}

float ContinuousTest::test_cart_pole(Coeus::NeuralNetwork& p_actor, Coeus::NeuralNetwork& p_critic, int p_episodes)
{
	Tensor action({ CartPole::ACTION }, Tensor::ZERO);
	Tensor state0({ CartPole::STATE }, Tensor::ZERO);
	Tensor critic_input({ CartPole::STATE + CartPole::ACTION }, Tensor::ZERO);

	_cart_pole.reset();
	copy_state(_cart_pole.get_state(true), state0);

	float total_reward = 0;
	int total_steps = 0;

	//printf("CartPole test...\n");

	for (int e = 0; e < p_episodes; ++e) {
		
		p_actor.activate(&state0);		
		action[0] = p_actor.get_output()->at(0);
		_cart_pole.perform_action(action[0]);

		/*
		critic_input.reset_index();
		critic_input.push_back(&state0);
		critic_input.push_back(&action);
		p_critic.activate(&critic_input);
		
		cout << p_actor.get_output()->at(0) << " " << p_critic.get_output()->at(0) << " state: " << state0 << " reward: " << _cart_pole.get_reward() << endl;
		*/
		
		copy_state(_cart_pole.get_state(true), state0);

		total_reward += _cart_pole.get_reward();
		total_steps += 1;

		if (_cart_pole.is_finished()) {
			break;
		}
	}
	printf("CartPole test finished in %i steps with reward %0.2f\n", total_steps, total_reward);

	return total_reward;
}

float ContinuousTest::test_mountain_car(Coeus::NeuralNetwork& p_actor, Coeus::LoggerInstance* p_logger)
{
	Tensor action({_mountain_car.ACTION_DIM()}, Tensor::ZERO);
	Tensor state0;
	int total_steps = 0;

	_mountain_car.reset();
	state0 = _mountain_car.get_state();

	while (!_mountain_car.is_finished()) {
		p_actor.activate(&state0);
		action[0] = p_actor.get_output()->at(0);
		_mountain_car.do_action(action);
		total_steps++;

		if (p_logger != nullptr)
		{
			string record = to_string(state0[0]) + ";" + to_string(state0[1]) + ";" + to_string(action[0]);
			p_logger->log(record);
		}

		state0 = _mountain_car.get_state();
	}
	
	printf("test finished in %i steps with reward %0.2f\n", total_steps, _mountain_car.get_reward());

	return _mountain_car.get_reward();
}

float ContinuousTest::evaluate_cart_pole(float p_reward)
{
	if (_rewards.size() == 100)
	{
		_rewards.erase(_rewards.begin());
	}
	_rewards.push_back(p_reward);

	float r_sum = 0;
	
	for(float r : _rewards)
	{
		r_sum += r;
	}

	r_sum /= 100;

	printf("CartPole evaluation with average reward %0.4f\n", r_sum);

	return r_sum;
}

void ContinuousTest::test_critic(NeuralNetwork& p_network)
{
	for(int i = 0; i < 11; i++)
	{
		
	}
}

void ContinuousTest::copy_state(vector<float>& p_observation, Tensor &p_state)
{
	for (int i = 0; i < p_observation.size(); i++)
	{
		p_state.set(i, p_observation[i]);
	}
}