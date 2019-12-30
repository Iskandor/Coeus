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
{
}

void MotivationTest::test1(const int p_episodes)
{
	const int hidden = 64;
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

	const ActorCritic agent(&network_critic, ADAM_RULE, 1e-3f, 0.99f, &network_actor, ADAM_RULE, 1e-4f);

	NeuralNetwork network_head;

	network_head.add_layer(new CoreLayer("hidden0", hidden * 4, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze.STATE_DIM()));
	network_head.add_layer(new CoreLayer("hidden1", hidden * 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections	
	network_head.add_connection("hidden0", "hidden1");
	network_head.init();

	NeuralNetwork network_forward_model;
	network_forward_model.add_layer(new CoreLayer("fm_input", hidden * 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), network_head.get_output_dim() + _maze.ACTION_DIM()));
	network_forward_model.add_layer(new CoreLayer("fm_output", _maze.STATE_DIM(), SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_forward_model.add_connection("fm_input", "fm_output");
	network_forward_model.init();
	
	NeuralNetwork network_inverse_model;
	network_inverse_model.add_layer(new CoreLayer("im_input", hidden * 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), network_head.get_output_dim() + network_head.get_output_dim()));
	network_inverse_model.add_layer(new CoreLayer("im_output", _maze.ACTION_DIM(), SOFTMAX, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_inverse_model.add_connection("im_input", "im_output");
	network_inverse_model.init();

	ICM icm(&network_forward_model, &network_inverse_model, &network_head, RADAM_RULE, 1e-3f);
	
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
			const float ri = icm.get_intrinsic_reward(&state0, &action, &state1, 0.1f);
			cri += ri;
			//cout << ri << endl;
			const float re = _maze.get_reward();
			const float reward = re + ri;
			agent.train(&state0, &action, &state1, reward, _maze.is_finished());
			icm.train(&state0, &action, &state1);
			//icm.add(&state0, &action, &state1);
			//icm.train(64);

			state0.override(&state1);
			step++;
		}

		if (_maze.is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		if (e % 100 == 0)
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
			Encoder::one_hot(s0, RandomGenerator::get_instance().random(0, _maze.STATE_DIM()));
			Encoder::one_hot(a, RandomGenerator::get_instance().random(0, _maze.ACTION_DIM()));
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
