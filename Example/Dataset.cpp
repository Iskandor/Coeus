//
// Created by user on 5. 11. 2017.
//

#include <fstream>
#include <algorithm>
#include "Dataset.h"
#include <string>

using namespace MNS;

Dataset::Dataset() {

}

Dataset::~Dataset() {
    for(auto it = _buffer.begin(); it != _buffer.end(); ++it) {
        for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
            delete it2->second;
        }
    }
}

void Dataset::loadData(string p_filename_v, string p_filename_m) {
    string line;

    ifstream v_file(p_filename_v);
    vector<string> v_lines;

    if (v_file.is_open())
    {
        while ( getline (v_file, line) )
        {
            v_lines.push_back(line);
        }
        v_file.close();
    }

    ifstream m_file(p_filename_m);
    vector<string> m_lines;

    if (m_file.is_open())
    {
        while ( getline (m_file, line) )
        {
            m_lines.push_back(line);
        }
        m_file.close();
    }

    parseLines(v_lines, m_lines);

    for(auto it = _buffer.begin(); it != _buffer.end(); ++it) {
        for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
            _permBuffer.push_back(it2->second);
        }
    }
}

void Dataset::parseLines(vector<string> p_vLines, vector<string> p_mLines) {
    int step0 = 0;
    vector<string> tokens;

    for(int i = 0; i < p_mLines.size(); i++) {
        tokens.clear();
        size_t pos = 0;

	    while ((pos = p_mLines[i].find(";")) != std::string::npos) {
            string token = p_mLines[i].substr(0, pos);
            tokens.push_back(token);
            p_mLines[i].erase(0, pos + 1);
        }

        int s = stoi(tokens[0]);
        int g = stoi(tokens[1]);
        int step1 = stoi(tokens[2]);

        if (step0 == 0 || step0 > step1 ) {
            _buffer[s][g] = new Sequence(s,g);
        }

		Tensor *data = new Tensor({ static_cast<int>(tokens.size() - 3) }, Tensor::ZERO);

        for(int t = 3; t < (int) (tokens.size()); t++) {
            data->set(t - 3, stod(tokens[t]));
        }

        _buffer[s][g]->addMotorData(data);

        step0 = step1;
    }

    for(int i = 0; i < p_vLines.size(); i++) {
        tokens.clear();
        size_t pos = 0;

	    while ((pos = p_vLines[i].find(";")) != std::string::npos) {
            string token = p_vLines[i].substr(0, pos);
            tokens.push_back(token);
            p_vLines[i].erase(0, pos + 1);
        }

        int s = stoi(tokens[0]);
        int g = stoi(tokens[1]);
        int p = stoi(tokens[2]);
        //int step1 = stoi(tokens[3]);

		Tensor *data = new Tensor({ static_cast<int>(tokens.size() - 4) }, Tensor::ZERO);

        //cout << tokens.size() - 4 << endl;

        for(int t = 4; t < (int) (tokens.size()); t++) {
            data->set(t - 4, stod(tokens[t]));
        }

        _buffer[s][g]->addVisualData(p, data);
    }
}

vector<Sequence*>* Dataset::permute() {
    //random_shuffle(_permBuffer.begin(), _permBuffer.end());
    return &_permBuffer;
}


