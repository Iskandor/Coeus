//
// Created by mpechac on 10. 3. 2016.
//

#include <iostream>
#include <chrono>
#include "Dataset.h"
#include <SOM.h>
#include <SOM_learning.h>

using namespace Coeus;

void sampleSOM() {
    Dataset dataset;
    DatasetConfig config = {13, 1, ",", 13};
    dataset.load("../data/wine.dat", config);
    dataset.normalize();

    SOM somNetwork(13, 8, 8, NeuralGroup::SIGMOID);
	SOM_learning learner(&somNetwork);
    double epochs = 2000;
	learner.init_training(0.01, epochs);

    for(int t = 0; t < epochs; t++) {
        dataset.permute();

		auto start = chrono::system_clock::now();

        for(int i = 0; i < dataset.getData()->size(); i++) {
			learner.train(dataset.getData()->at(i).first);
        }

		auto end = chrono::system_clock::now();
		chrono::duration<double> elapsed_seconds = end - start;
		cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
        cout << t << " qError: " << learner.analyzer()->q_error() << " WD: " << learner.analyzer()->winner_diff() << endl;
		learner.param_decay();
    }
}