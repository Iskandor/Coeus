#pragma once
namespace Coeus
{
	class __declspec(dllexport) ILearningRateModule
	{
	public:
		ILearningRateModule();
		virtual ~ILearningRateModule();

		virtual double get_alpha() = 0;
	};
}