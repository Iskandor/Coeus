# Coeus

Lightweight C++ Library for reinforcement learning algorithms

![](https://raw.githubusercontent.com/Iskandor/Coeus/master/Logo/logo.jpg)

### Dependencies
Intel Math Kernel Library (https://software.intel.com/en-us/mkl)
ZLIB (https://github.com/madler/zlib)

### Features
**Neural Networks**
- Feed-forward neural networks
- SGD, ADAM, RADAM optimizers

**Reinforcement learning algorithms**
- TD-learning
- Q-Learning
- SARSA
- Deep Q-Learning

**Continuous reinforcement learning algorithms**
- CACLA
- DDPG
 
**Motivation models**
- Predictive error model
- Surprise model
 
**Other features**
- Simple model building demanding only few lines of code
- Save/Load in numpy array format (CNPy library)

### Example XOR problem
```cpp
#include "neural_network.h"
#include "loss_functions.h"
#include "tensor_initializer.h"
#include "adam.h"

int main()
{
	float input_data[8] = {0,0,0,1,1,0,1,1};
	float target_data[4] = { 0,1,1,0 };

	tensor input({ 4, 2 }, input_data);
	tensor target({ 4, 1 }, target_data);

	neural_network network;

	network.add_layer(new dense_layer("hidden0", 8, activation_function::sigmoid(), tensor_initializer::lecun_uniform(), { 2 }));
	network.add_layer(new dense_layer("hidden1", 4, activation_function::sigmoid(), tensor_initializer::lecun_uniform()));
	network.add_layer(new dense_layer("output", 1, activation_function::sigmoid(), tensor_initializer::lecun_uniform()));
	network.add_connection("hidden0", "hidden1");
	network.add_connection("hidden1", "output");
	network.init();

	mse_function loss;
	adam optimizer(&network, 1e-2f);

	for (int t = 0; t < 500; t++) {
		tensor& output = network.forward(&input);
		const float error = loss.forward(output, target);
		network.backward(loss.backward(output, target));
		optimizer.update();

		cout << "Episode " << t;
		cout << " error: " << error << endl;
	}

	cout << network.forward(&input) << endl;
}
```

### Planned features
- AC,AC2,AC3
- PPO
- convolutional networks
- ALE support
- MuJoCo support
 
### Author
Matej Pechac is doctoral student of informatics specializing in the area of reinforcement learning and intrinsic motivation
- univeristy webpage: http://dai.fmph.uniba.sk/w/Matej_Pechac/en
- contact: matej.pechac@gmail.com