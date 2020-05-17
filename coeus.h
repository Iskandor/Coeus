#pragma once

#ifdef _WIN64
#define COEUS_DLL_API __declspec(dllexport)
#else
#define COEUS_DLL_API 
#endif