#include "IrisTest.h"
#include "LSOM_params.h"
#include "LSOM_learning.h"
#include "SOM_learning.h"


IrisTest::IrisTest() {
	_lsom = nullptr;
	_dataset.load_data("./data/iris.data");
}

IrisTest::~IrisTest()
{
	delete _lsom;
}

void IrisTest::init() {
	_lsom = new LSOM("LSOM", 4, 4, 4, NeuralGroup::EXPONENTIAL);
}

void IrisTest::run(const int p_epochs) {
	SOM_analyzer analyzer;
	LSOM_params params(_lsom);
	params.init_training(0.1, 0.1, p_epochs);
	LSOM_learning learner(_lsom, &params, &analyzer);

	vector<IrisDatasetItem>* data = nullptr;

	for(int t = 0; t < p_epochs; t++) {
		cout << "Epoch " << t << endl;
		
		data = _dataset.permute();

		const auto start = chrono::system_clock::now();

		for(int i = 0; i < data->size(); i++) {
			//cout << data->at(i).target << endl;
			learner.train(data->at(i).data);
		}

		const auto end = chrono::system_clock::now();
		chrono::duration<double> elapsed_seconds = end - start;
		cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
		cout << " LSOM qError: " << analyzer.q_error() << " WD: " << analyzer.winner_diff(_lsom->get_lattice()->get_dim()) << endl;

		analyzer.end_epoch();
		params.param_decay();
	}	
}
