#include "PID.h"
#include <uWS/uWS.h>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	this->Kp = Kp;
	this->Ki = Ki;
	this->Kd = Kd;
	
	p_error = 0;
	i_error = 0;
	d_error = 0;
	
	error_squared = 0;
	
	steps = 1;
	best_error = std::numeric_limits<double>::max();
	n_wait_steps = 100;
	n_eval_steps = 2000;
}

void PID::UpdateError(double cte) {
	d_error = cte - p_error;
	i_error += cte;
	p_error = cte;
	
}

double PID::TotalError(double speed_factor) {
	return (-(Kp*p_error*speed_factor) - (Kd*d_error) - (Ki*i_error*(1-speed_factor)));
}

void PID::Reset(uWS::WebSocket<uWS::SERVER> ws){
	string reset_msg = "42[\"reset\",{}]";
	ws.send(reset_msg.data(), reset_msg.length(), uWS::OpCode::TEXT);
}

void PID::Twiddle(){
	if(steps >= n_wait_steps){
		double p[3] = {0, 0, 0};
		double dp[3] = {1, 1, 1};

		// check how to reset
		//Reset(uWS::WebSocket<uWS::SERVER> ws);

		for(int i=0; i<3; i++){
			p[i] += dp[i];

			// how to reset
			// how to run and get error
			if(avg_error < best_error){
				cout << "Improvement" << endl;
				best_error = avg_error;
				dp[i] *= 1.1;
			}
			else {
				p[i] -= 2 * dp[i];
				// how to reset
				// how to run and get error
				if(avg_error < best_error){
					best_error = avg_error;
					dp[i] *= 1.1;
				}
				else {
					p[i] += dp[i];
					dp[i] *= 0.9;
				}
			}
		}
		this->Kp = p[0];
		this->Ki = p[1];
		this->Kd = p[2];
	}
}

