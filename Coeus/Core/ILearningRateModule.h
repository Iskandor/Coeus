#pragma once
namespace Coeus
{
	class __declspec(dllexport) ILearningRateModule
	{
	public:
		ILearningRateModule();
		virtual ~ILearningRateModule();

		virtual float get_alpha() = 0;
	};
}