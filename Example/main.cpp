
#include "FFN.h"
#include "IrisTest.h"
#include "RNN.h"
#include "MazeExample.h"
#include "Encoder.h"
#include <bitset>
#include "mnist_reader.hpp"


using namespace std;

int main()
{
	const string MNIST_DATA_LOCATION = "./data/";
	// MNIST_DATA_LOCATION set by MNIST cmake config
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

	// Load MNIST data
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

	std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
	std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
	std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
	std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

	//FFN model;
	//model.run();
	//model.run_iris();

	//RNN model;
	//model.run_pack();
	//model.test_pack();
	//model.test_pack_cm();
	//model.test_pack_alt();
	//model.run_add_problem();
	
	//MazeExample example;
	//example.example_q(64, 1e-3, 0, true);
	//example.example_sarsa(64, 1e-3, 0, true);
	//example.example_actor_critic(64);
	//example.example_deep_q(64, 1e-3, 0, true);
	//example.example_icm(64);

	/*
	IrisTest iris;

	iris.init();
	iris.run(2000);
	iris.test();
	*/

	system("pause");

	return 0;
}
