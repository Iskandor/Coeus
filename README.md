# Coeus

Lightweight C++ Library supporting deep and reinforcement learning algorithms

![](https://raw.githubusercontent.com/Iskandor/Coeus/master/Logo/logo.jpg)

### Dependencies
Intel Math Kernel Library (https://software.intel.com/en-us/mkl)

### Features
**Neural Networks**
- Feed-forward neural networks and recurrent neural networks (Elman, Jordna, LSTM)
- Self-organizing maps (SOM) and recurrent self-organizing maps
- Convolutional neural networks

**Reinforcement learning algorithms**
- TD-learning
- Q-Learning
- SARSA
- Double Q-learning
- Deep Q-Learning
- Actor-Critic algorithms (AC, A2C)

**Continuous reinforcement learning algorithms**
- CACLA
 
**Other features**
- BLAS backend
- Supports parallelization across samples using OpenMP library
- Simple model building demanding only few lines of code

### Example XOR problem
```cpp
	float data_i[8]{ 0,0,0,1,1,0,1,1 };
	float data_t[4]{ 0,1,1,0 };
	Tensor input({ 4, 2 }, Tensor::ZERO);
	Tensor target({ 4, 1 }, Tensor::ZERO);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 2; j++)
		{
			input.set(i, j, data_i[i * 2 + j]);
		}
		target.set(i, 0, data_t[i]);
	}
	
	NeuralNetwork network;

	network.add_layer(new CoreLayer("hidden", 4, SIGMOID, new TensorInitializer(LECUN_UNIFORM), 2));
	network.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(LECUN_UNIFORM)));
	network.add_connection("hidden", "output");

	network.init();

	BackProp optimizer(&network);
	optimizer.init(new QuadraticCost(), 0.5f /*lr*/, 0.9f /*momentum*/, true /*nesterov momentum*/);
	
	for (int t = 0; t < 1000; t++) {
		float error = optimizer.train(&input, &target);
		cout << "Error: " << error << endl;
	}
```

### Planned features
- Residual networks
- Gated recurrent unit
- DDPG, NAC, TRPO, PPO, CACER
- CUDA support
- Intrinsic motivation modules
- ALE interface and examples
 


### Author
Matej Pechac is doctoral student of informatics specializing in the area of reinforcement learning and intrinsic motivation
- univeristy webpage: http://dai.fmph.uniba.sk/w/Matej_Pechac/en
- contact: matej.pechac@gmail.com