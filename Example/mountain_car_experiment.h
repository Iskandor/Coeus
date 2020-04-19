#pragma once
#include "simple_continuous_env.h"
#include "mountain_car.h"

class mountain_car_experiment
{
public:
	mountain_car_experiment();
	~mountain_car_experiment();

	void simple_ddpg(int p_epochs);
	void simple_cacla(int p_epochs);
	void simple_dqn(int p_epochs);

	void run_ddpg(int p_epochs);
	void run_ddpg_fm(int p_epochs);

private:
	simple_continuous_env _simple_env;
	mountain_car _env;
};

