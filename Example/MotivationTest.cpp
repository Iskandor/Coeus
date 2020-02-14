#include "MotivationTest.h"
#include "Encoder.h"
#include "ActorCritic.h"
#include "TensorInitializer.h"
#include "CoreLayer.h"
#include "RandomGenerator.h"
#include "ICM.h"
#include "ADAM.h"
#include "RAdam.h"
#include "BackProph.h"
#include "CartPole.h"
#include "DDPG.h"
#include "ForwardModel.h"
#include "Logger.h"
#include "ContinuousExploration.h"

MotivationTest::MotivationTest()
{
	/*
	int topology[] =
	{	0, 0, 0, 0,
		0, 2, 0, 2,
		0, 0, 0, 2,
		2, 0, 0, 0 };

	_maze.init(topology, 4, 4, 15, false);
	*/

	int topology[] =
	{	0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 2, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 2, 0, 0,
		0, 0, 0, 2, 0, 0, 0, 0,
		0, 2, 2, 0, 0, 0, 2, 0,
		0, 2, 0, 0, 2, 0, 2, 0,
		0, 0, 0, 2, 0, 0, 0, 0
	};

	_maze.init(topology, 8, 8, 63, false);
}

MotivationTest::~MotivationTest()
= default;

void MotivationTest::cart_pole_icm(int p_episodes)
{
	CartPole env;

	const int hidden = 128;
	const float limit = 0.01f;

	NeuralNetwork network_features;

	network_features.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), CartPole::STATE));
	network_features.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections	
	network_features.add_connection("hidden0", "hidden1");
	network_features.init();

	NeuralNetwork network_forward_model;
	network_forward_model.add_layer(new CoreLayer("fm_input", hidden / 4, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), network_features.get_output_dim() + CartPole::ACTION));
	network_forward_model.add_layer(new CoreLayer("fm_output", CartPole::STATE, TANH, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_forward_model.add_connection("fm_input", "fm_output");
	network_forward_model.init();

	NeuralNetwork network_inverse_model;
	network_inverse_model.add_layer(new CoreLayer("im_input", hidden / 4, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), network_features.get_output_dim() + network_features.get_output_dim()));
	network_inverse_model.add_layer(new CoreLayer("im_output", CartPole::ACTION, TANH, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_inverse_model.add_connection("im_input", "im_output");
	network_inverse_model.init();

	ICM icm(&network_forward_model, &network_inverse_model, &network_features, RADAM_RULE, 1e-4f);

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), CartPole::STATE + CartPole::ACTION));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), CartPole::STATE));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_actor.add_layer(new CoreLayer("output", CartPole::ACTION, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	DDPG agent(&network_critic, RADAM_RULE, 1e-3f, 0.99f, &network_actor, RADAM_RULE, 1e-4f, 10000, 64);

	Tensor action({ CartPole::ACTION }, Tensor::ZERO);
	Tensor state0({ CartPole::STATE }, Tensor::ZERO);
	Tensor state1({ CartPole::STATE }, Tensor::ZERO);

	const float sigma = 1.f;
	ContinuousExploration exploration;
	exploration.init_gaussian(sigma);
	
	for (int e = 0; e < p_episodes; ++e) {
		//printf("CartPole episode %i...\n", e);
		float cri = 0;
		float cre = 0;
		int total_steps = 0;
		
		env.reset();

		copy_state(env.get_state(true), state0);


		while (true) {
			action = exploration.explore(agent.get_action(&state0));

			env.perform_action(action[0]);
			copy_state(env.get_state(true), state1);

			total_steps++;

			const float ri = icm.get_intrinsic_reward(&state0, &action, &state1, .01f);
			cri += ri;
			const float re = env.get_reward();
			cre += re;
			const float reward = re + ri;
			agent.train(&state0, &action, &state1, reward, env.is_finished());
			icm.train(&state0, &action, &state1);
			//icm.add(&state0, &action, &state1);
			//icm.train(64);

			
			state0 = state1;

			if (e % 1000 == 0) {
				//cout << network_critic.get_output()->at(0) << " " << delta << endl;
			}

			if (env.is_finished()) {
				break;
			}
		}
		
		printf("CartPole ICM Episode %i internal reward %0.4f ", e, cri);

		if (evaluate_cart_pole(cre))
		{
			break;
		}
	}

	//test_cart_pole(network_actor, network_critic, 6000);
}

void MotivationTest::cart_pole_icm2(int p_episodes, bool p_log)
{
	_rewards.clear();
	CartPole env;

	const int hidden = 128;
	const float limit = 0.01f;

	NeuralNetwork network_forward_model;
	network_forward_model.add_layer(new CoreLayer("fm_input", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), CartPole::STATE + CartPole::ACTION));
	network_forward_model.add_layer(new CoreLayer("fm_output", CartPole::STATE, TANH, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_forward_model.add_connection("fm_input", "fm_output");
	network_forward_model.init();

	ForwardModel forward_model(&network_forward_model, ADAM_RULE, 1e-3f);

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), CartPole::STATE + CartPole::ACTION));
	network_critic.add_layer(new CoreLayer("hidden1", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_critic.add_layer(new CoreLayer("hidden2", hidden / 2, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "hidden2");
	network_critic.add_connection("hidden2", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), CartPole::STATE));
	network_actor.add_layer(new CoreLayer("hidden1", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_actor.add_layer(new CoreLayer("hidden2", hidden / 2, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_actor.add_layer(new CoreLayer("output", CartPole::ACTION, TANH, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "hidden2");
	network_actor.add_connection("hidden2", "output");
	network_actor.init();

	DDPG agent(&network_critic, ADAM_RULE, 1e-4f, 0.99f, &network_actor, ADAM_RULE, 1e-4f, 10000, 64);

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
		float cri = 0;
		float cre = 0;
		int total_steps = 0;
		const float total_reward = test_cart_pole(env, network_actor, network_critic, 195);

		env.reset();

		copy_state(env.get_state(true), state0);


		while (true) {
			action = exploration.explore(agent.get_action(&state0));

			env.perform_action(action[0]);
			copy_state(env.get_state(true), state1);

			total_steps++;

			const float ri = forward_model.train(&state0, &action, &state1);
			cri += ri;
			const float re = env.get_reward();
			cre += re;
			const float reward = re + ri;
			agent.train(&state0, &action, &state1, reward, env.is_finished());

			//cout << *network_forward_model.get_output() << " " << state1 << endl;

			state0 = state1;

			if (env.is_finished()) {
				break;
			}
		}

		printf("CartPole ICM Episode %i internal reward %0.4f ", e, cri);

		const float avg_reward = evaluate_cart_pole(total_reward);
		if (p_log) logger.log(to_string(avg_reward) + ";" + to_string(cri));
	}

	if (p_log) logger.close();
	//test_cart_pole(network_actor, network_critic, 6000);	
}

void MotivationTest::test_icm(const int p_episodes)
{
	const int hidden = 128;
	const float limit = 0.01f;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze.STATE_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_critic.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze.STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_actor.add_layer(new CoreLayer("output", _maze.ACTION_DIM(), SOFTMAX, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	const ActorCritic agent(new TD(&network_critic, ADAM_RULE, 1e-3f, 0.99f), &network_actor, ADAM_RULE, 1e-4f);

	NeuralNetwork network_feature;

	network_feature.add_layer(new CoreLayer("hidden0", hidden * 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze.STATE_DIM()));
	network_feature.add_layer(new CoreLayer("hidden1", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections	
	network_feature.add_connection("hidden0", "hidden1");
	network_feature.init();

	NeuralNetwork network_forward_model;
	network_forward_model.add_layer(new CoreLayer("fm_input", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), network_feature.get_output_dim() + _maze.ACTION_DIM()));
	network_forward_model.add_layer(new CoreLayer("fm_output", _maze.STATE_DIM(), SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_forward_model.add_connection("fm_input", "fm_output");
	network_forward_model.init();
	
	NeuralNetwork network_inverse_model;
	network_inverse_model.add_layer(new CoreLayer("im_input", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), network_feature.get_output_dim() + network_feature.get_output_dim()));
	network_inverse_model.add_layer(new CoreLayer("im_output", _maze.ACTION_DIM(), SOFTMAX, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_inverse_model.add_connection("im_input", "im_output");
	network_inverse_model.init();

	ICM icm(&network_forward_model, &network_inverse_model, &network_feature, RADAM_RULE, 1e-4f, 100000);
	
	train_icm(icm);
	system("pause");
		
	Tensor state0, state1;
	Tensor action({ _maze.ACTION_DIM() }, Tensor::ZERO);

	int wins = 0, loses = 0;
	

	//Logger::instance().init("log.log");

	for (int e = 0; e < p_episodes; e++) {
		float cri = 0;
		int step = 0;
		_maze.reset();
		state0 = _maze.get_state();

		network_critic.activate(&state0);

		while (!_maze.is_finished()) {
			network_actor.activate(&state0);

			const int action0 = RandomGenerator::get_instance().choice(network_actor.get_output()->arr(), _maze.ACTION_DIM());
			Encoder::one_hot(action, action0);
			_maze.do_action(action);
			state1 = _maze.get_state();
			const float ri = icm.get_intrinsic_reward(&state0, &action, &state1, 0.01f);
			cri += ri;
			//cout << ri << endl;
			const float re = _maze.get_reward();
			const float reward = re + ri;
			agent.train(&state0, &action, &state1, reward, _maze.is_finished());
			//icm.train(&state0, &action, &state1);
			icm.add(&state0, &action, &state1);
			icm.train(64);

			state0.override(&state1);
			step++;
		}

		if (_maze.is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		if (e % 100 == 0 && false)
		{
			test_policy(network_actor);
			cout << endl;
			test_v(&network_critic);
			cout << endl;
			//test_icm_model2(icm, network_actor);
			test_icm_model(icm);
			system("pause");
		}

		printf("Actor-Critic Episode %i results: %i / %i\n", e, wins, loses);
		printf("%f\n", cri / step);
	}

	test_policy(network_actor);
	cout << endl;
	test_v(&network_critic);
	cout << endl;
	test_icm_model(icm);
}

void MotivationTest::test_gm2(int p_episodes)
{
	const int hidden = 128;
	const float limit = 0.01f;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze.STATE_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_critic.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze.STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_actor.add_layer(new CoreLayer("output", _maze.ACTION_DIM(), SOFTMAX, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	const ActorCritic agent(new TD(&network_critic, ADAM_RULE, 1e-3f, 0.99f), &network_actor, ADAM_RULE, 1e-4f);

	NeuralNetwork network_autoencoder;

	network_autoencoder.add_layer(new CoreLayer("hidden0", _maze.STATE_DIM() * 16, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze.STATE_DIM()));
	network_autoencoder.add_layer(new CoreLayer("hidden1", _maze.STATE_DIM() * 8, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_autoencoder.add_layer(new CoreLayer("latent", _maze.STATE_DIM() / 4, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_autoencoder.add_layer(new CoreLayer("hidden2", _maze.STATE_DIM() * 8, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_autoencoder.add_layer(new CoreLayer("hidden3", _maze.STATE_DIM() * 16, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_autoencoder.add_layer(new CoreLayer("output", _maze.STATE_DIM(), SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections	
	network_autoencoder.add_connection("hidden0", "hidden1");
	network_autoencoder.add_connection("hidden1", "latent");
	network_autoencoder.add_connection("latent", "hidden2");
	network_autoencoder.add_connection("hidden2", "hidden3");
	network_autoencoder.add_connection("hidden3", "output");
	network_autoencoder.init();

	GM2 gm2(&network_autoencoder, RADAM_RULE, 1e-4f, 1e5);

	train_gm2(gm2);
	system("pause");

	Tensor state0, state1;
	Tensor action({ _maze.ACTION_DIM() }, Tensor::ZERO);

	int wins = 0, loses = 0;


	//Logger::instance().init("log.log");

	for (int e = 0; e < p_episodes; e++) {
		float cri = 0;
		int step = 0;
		_maze.reset();
		state0 = _maze.get_state();

		network_critic.activate(&state0);

		while (!_maze.is_finished()) {
			network_actor.activate(&state0);

			const int action0 = RandomGenerator::get_instance().choice(network_actor.get_output()->arr(), _maze.ACTION_DIM());
			Encoder::one_hot(action, action0);
			_maze.do_action(action);
			state1 = _maze.get_state();
			const float ri = gm2.uncertainty_motivation(&state1);
			cri += ri;
			//cout << ri << endl;
			const float re = _maze.get_reward();
			const float reward = re + ri;
			agent.train(&state0, &action, &state1, reward, _maze.is_finished());
			gm2.train(&state1);

			state0.override(&state1);
			step++;
		}

		if (_maze.is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		if (e % 100 == 0 && false)
		{
			test_policy(network_actor);
			cout << endl;
			test_v(&network_critic);
			cout << endl;
			//test_icm_model2(icm, network_actor);
			test_gm2_model(gm2);
			system("pause");
		}

		printf("Actor-Critic Episode %i results: %i / %i\n", e, wins, loses);
		printf("%f\n", cri / step);
	}

	test_policy(network_actor);
	cout << endl;
	test_v(&network_critic);
	cout << endl;
	test_gm2_model(gm2);
}

void MotivationTest::train_icm(ICM& p_icm)
{
	Tensor s0 = Tensor::Zero({ _maze.STATE_DIM() });
	Tensor s1 = Tensor::Zero({ _maze.STATE_DIM() });
	Tensor a = Tensor::Zero({ _maze.ACTION_DIM() });

	for(int i = 0; i < 2e3; i++)
	{
		float error = 0;
		for(int j = 0; j < 25; j++)
		{
			int is0 = RandomGenerator::get_instance().random(0, _maze.STATE_DIM() - 1);
			int ia = RandomGenerator::get_instance().random(0, _maze.ACTION_DIM() - 1);			
			Encoder::one_hot(s0, is0);
			Encoder::one_hot(a, ia);			
			_maze.set_state(s0);
			_maze.do_action(a);
			s1 = _maze.get_state();

			//p_icm.add(&s0, &a, &s1);
			error += p_icm.train(&s0, &a, &s1);
			//error += p_icm.train(64);
		}

		printf("Episode %i error %1.6f\n", i, error);
	}

	test_icm_model(p_icm);
}

void MotivationTest::train_gm2(GM2& p_gm2)
{
	Tensor s0 = Tensor::Zero({ _maze.STATE_DIM() });
	Tensor s1 = Tensor::Zero({ _maze.STATE_DIM() });
	Tensor a = Tensor::Zero({ _maze.ACTION_DIM() });

	for (int i = 0; i < 4000; i++)
	{
		float error = 0;
		for (int j = 0; j < 25; j++)
		{
			int is0 = RandomGenerator::get_instance().random(0, _maze.STATE_DIM() - 1);
			int ia = RandomGenerator::get_instance().random(0, _maze.ACTION_DIM() - 1);
			Encoder::one_hot(s0, is0);
			Encoder::one_hot(a, ia);
			_maze.set_state(s0);
			_maze.do_action(a);
			s1 = _maze.get_state();

			p_gm2.add(&s0, &a, &s1);
			error += p_gm2.train(64);
		}

		printf("Episode %i error %1.6f\n", i, error);
	}

	test_gm2_model(p_gm2);
}

void MotivationTest::test_icm_model(ICM& p_icm)
{
	Tensor s0 = Tensor::Zero({ _maze.STATE_DIM() });
	Tensor s1 = Tensor::Zero({ _maze.STATE_DIM() });
	Tensor a = Tensor::Zero({ _maze.ACTION_DIM() });
	

	for (unsigned int i = 0; i < _maze.mazeY(); i++)
	{
		for (unsigned int j = 0; j < _maze.mazeX(); j++)
		{
			Encoder::one_hot(s0, i * _maze.mazeX() + j);
			float ri = 0;
			
			for(int ai = 0; ai < _maze.ACTION_DIM(); ai++)
			{
				Encoder::one_hot(a, ai);
				_maze.set_state(s0);
				_maze.do_action(a);
				s1 = _maze.get_state();
				ri += p_icm.get_intrinsic_reward(&s0, &a, &s1);
			}
			
			ri /= _maze.ACTION_DIM();
			
			printf("%1.2f ", ri);
		}

		cout << endl;
	}

	cout << endl;
	
}

void MotivationTest::test_gm2_model(GM2& p_gm2) const
{
	Tensor s = Tensor::Zero({ _maze.STATE_DIM() });

	for (unsigned int i = 0; i < _maze.mazeY(); i++)
	{
		for (unsigned int j = 0; j < _maze.mazeX(); j++)
		{
			Encoder::one_hot(s, i * _maze.mazeX() + j);
			const float ri = p_gm2.uncertainty_motivation(&s);			

			//printf("%1.2f ", ri);
		}

		cout << endl;
	}

	cout << endl;
}

void MotivationTest::test_icm_model2(ICM& p_icm, NeuralNetwork& p_actor)
{
	Tensor s0 = Tensor::Zero({ _maze.STATE_DIM() });
	Tensor s1 = Tensor::Zero({ _maze.STATE_DIM() });
	Tensor a = Tensor::Zero({ _maze.ACTION_DIM() });


	for (unsigned int i = 0; i < _maze.mazeY(); i++)
	{
		for (unsigned int j = 0; j < _maze.mazeX(); j++)
		{
			Encoder::one_hot(s0, i * _maze.mazeX() + j);
			p_actor.activate(&s0);
			Encoder::one_hot(a, p_actor.get_output()->max_value_index());
			
			_maze.set_state(s0);
			_maze.do_action(a);
			s1 = _maze.get_state();
			
			float ri = p_icm.get_intrinsic_reward(&s0, &a, &s1);
			printf("%1.2f ", ri);
		}

		cout << endl;
	}

	cout << endl;
}

void MotivationTest::test_policy(NeuralNetwork &p_network)
{
	_maze.reset();

	vector<float> sensors;
	Tensor action = Tensor::Zero({ _maze.ACTION_DIM() });
	Tensor state;

	string action_labels[4] = { "Up","Right","Down","Left" };
	int step = 0;

	while (!_maze.is_finished() && step < _maze.mazeY() + _maze.mazeX()) {
		cout << _maze.toString();
		state = _maze.get_state();
		p_network.activate(&state);

		for (int i = 0; i < p_network.get_output()->size(); i++) {
			printf("%s: %1.4f ", action_labels[i].c_str(), (*p_network.get_output())[i]);
		}
		printf(" -> Step %i (%s)\n", step, action_labels[p_network.get_output()->max_value_index()].c_str());

		const int action_index = p_network.get_output()->max_value_index();
		Encoder::one_hot(action, action_index);
		_maze.do_action(action);
		step++;	
	}
	cout << _maze.toString() << endl;
	

	char action_abrev[4] = { 'U','R','D','L' };
	
	Tensor s = Tensor::Zero({ _maze.STATE_DIM() });

	for (unsigned int i = 0; i < _maze.mazeY(); i++)
	{
		for (unsigned int j = 0; j < _maze.mazeX(); j++)
		{
			const int a = i * _maze.mazeX() + j;
			Encoder::one_hot(s, a);

			p_network.activate(&s);
			cout << action_abrev[p_network.get_output()->max_value_index()];
		}

		cout << endl;
	}

	cout << endl;
}

float MotivationTest::evaluate_cart_pole(const float p_reward)
{
	if (_rewards.size() == 100)
	{
		_rewards.erase(_rewards.begin());
	}
	_rewards.push_back(p_reward);

	float r_sum = 0;

	for (float r : _rewards)
	{
		r_sum += r;
	}

	r_sum /= _rewards.size();

	printf("CartPole evaluation with average reward %0.4f\n", r_sum);

	return r_sum;
}

float MotivationTest::test_cart_pole(CartPole& p_env, NeuralNetwork& p_actor, NeuralNetwork& p_critic, int p_episodes)
{
	Tensor action({ CartPole::ACTION }, Tensor::ZERO);
	Tensor state0({ CartPole::STATE }, Tensor::ZERO);
	Tensor critic_input({ CartPole::STATE + CartPole::ACTION }, Tensor::ZERO);

	p_env.reset();
	copy_state(p_env.get_state(true), state0);

	float total_reward = 0;
	int total_steps = 0;

	//printf("CartPole test...\n");

	for (int e = 0; e < p_episodes; ++e) {

		p_actor.activate(&state0);
		action[0] = p_actor.get_output()->at(0);
		p_env.perform_action(action[0]);

		/*
		critic_input.reset_index();
		critic_input.push_back(&state0);
		critic_input.push_back(&action);
		p_critic.activate(&critic_input);

		cout << p_actor.get_output()->at(0) << " " << p_critic.get_output()->at(0) << " state: " << state0 << " reward: " << _cart_pole.get_reward() << endl;
		*/

		copy_state(p_env.get_state(true), state0);

		total_reward += p_env.get_reward();
		total_steps += 1;

		if (p_env.is_finished()) {
			break;
		}
	}
	printf("CartPole test finished in %i steps with reward %0.2f\n", total_steps, total_reward);

	return total_reward;
}

void MotivationTest::copy_state(vector<float>& p_observation, Tensor& p_state)
{
	for (int i = 0; i < p_observation.size(); i++)
	{
		p_state.set(i, p_observation[i]);
	}
}

void MotivationTest::test_v(NeuralNetwork* p_network)
{
	_maze.reset();

	vector<float> sensors;
	Tensor action = Tensor::Zero({ _maze.ACTION_DIM() });
	Tensor state;

	Tensor s = Tensor::Zero({ _maze.STATE_DIM() });

	for (unsigned int i = 0; i < _maze.mazeY(); i++)
	{
		for (unsigned int j = 0; j < _maze.mazeX(); j++)
		{
			const int a = i * _maze.mazeX() + j;
			Encoder::one_hot(s, a);

			p_network->activate(&s);

			printf("%1.2f ", (*p_network->get_output())[0]);
		}

		cout << endl;
	}
}
