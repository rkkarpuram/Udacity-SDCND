#ifndef PID_H
#define PID_H

#include <uWS/uWS.h>

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;

  /*
  * Coefficients
  */ 
  double Kp;
  double Ki;
  double Kd;
	
	double prev_cte;
	double sum_cte;
	
	/*
	* Twiddle variables
	*/
	double error_squared;
	int steps;				// counter to count number of steps
	int param_idx;	// parameter index to refer specific gain
	int n_wait_steps;	// number of steps to wait before evaluation begins
	int n_eval_steps;	// max number of steps for evaluating error
	double best_error;
	double avg_error;	// average error that is returned to check with twiddle's best_error
	bool to_twiddle;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
	
	/*
	* Reset the simulator
	*/
	void Reset(uWS::WebSocket<uWS::SERVER> ws);	//uWS::WebSocket<uWS::SERVER> ws
	
	/*
	* Implement twiddle algorithm to optimize the gain parameters.
	*/
	void Twiddle();
	
};

#endif /* PID_H */
