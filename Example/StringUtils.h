//
// Created by mpechac on 10. 3. 2016.
//

#ifndef LIBNEURONET_STRINGUTILS_H
#define LIBNEURONET_STRINGUTILS_H

#include <string>
#include <vector>

using namespace std;

class StringUtils {
public:
    StringUtils() {};
    ~StringUtils() {};
    static string &ltrim(string &s);
    static string &rtrim(string &s);
    static string &trim(string &s);
    static vector<std::string> split(const std::string &s, char delim);
    static vector<std::string> &split(const std::string &s, char delim, vector<std::string> &elems);
};


#endif //LIBNEURONET_STRINGUTILS_H
