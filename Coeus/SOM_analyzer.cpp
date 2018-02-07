#include "SOM_analyzer.h"
#include <fstream>

using namespace Coeus;

SOM_analyzer::SOM_analyzer()
{
	_umatrix = nullptr;
	_q_error = 0;
}


SOM_analyzer::~SOM_analyzer()
{
	if (_umatrix != nullptr) {
		delete _umatrix;
	}
}

void SOM_analyzer::update(SOM* p_som, const int p_winner)
{
	_winner_set.insert(p_winner);
	_q_error += p_som->calc_distance(p_winner);
}

void Coeus::SOM_analyzer::create_umatrix(SOM * p_som)
{
	_umatrix = new Tensor({ p_som->dim_x(), p_som->dim_y() }, Tensor::ZERO);

	int pos, pos_x, pos_y;
	double s;

	for (int i = 0; i < p_som->dim_x() * p_som->dim_y(); i++) {
		p_som->get_position(i, pos_x, pos_y);
		s = 0;

		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				pos = p_som->get_position(pos_x + x, pos_y + y);

				if (pos != -1) {
					s += p_som->calc_distance(i, pos);
				}
			}
		}

		_umatrix->set(pos_y, pos_x, s);
	}
}

void SOM_analyzer::end_epoch()
{
	_winner_set.clear();
	_q_error = 0;
}

void SOM_analyzer::merge(vector<SOM_analyzer*> &p_analyzers) {
	for (auto it = p_analyzers.begin(); it != p_analyzers.end(); ++it) {
		_q_error += (*it)->_q_error;
		_winner_set.insert((*it)->_winner_set.begin(), (*it)->_winner_set.end());
		(*it)->end_epoch();
	}
}

double SOM_analyzer::winner_diff(const int p_size) const {
	return static_cast<double>(_winner_set.size()) / static_cast<double>(p_size);
}

void SOM_analyzer::save_umatrix(string p_filename)
{
	ofstream file;
	file.open(p_filename);
	file << _umatrix->shape(0) << endl;
	file << _umatrix->shape(1) << endl;

	for (int i = 0; i < _umatrix->size(); i++) {
		file << _umatrix->at(i);
		if (i < _umatrix->size() - 1) file << ",";
	}
	
	file.close();
}
