#ifndef INFLOW_H
#define INFLOW_H

#define MAX_NUM_INFLOWS 10 //max number of flowers in simulation

// Object used to add to fluid quantities
struct InFlowData
{
public:
	//which value to add to the fluid quantity
	enum QuantityToAdd
	{
		U_VEL, V_VEL, DENSITY, RED, GREEN, BLUE
	};

	InFlowData(float u = 0.0f, float v = 0.0f, float d = 0.0f,
		float r = 0.0f, float g = 0.0f, float b = 0.0f,
		float x0 = 0.0f, float y0 = 0.0f,
		float w = 0.0f, float h = 0.0f
	);

	//velocity of quantity and concentration
	double u_vel = 0.0f;
	double v_vel = 0.0f;
	double density = 0.0f;
	//0 - 255.0 color values
	double red = 0.0f;
	double blue = 0.0f;
	double green = 0.0f;
	//rectangle of influence
	double x = 0.0f, y = 0.0f;
	double width = 0.0f, height = 0.0f;
};


/*
class InFlow
{
public:
	
	InFlow(float u = 0.0f, float v = 0.0f, float d = 0.0f,
		float r = 0.0f, float b = 0.0f, float g = 0.0f,
		float x = 0.0f, float y = 0.0f,
		float w = 0.0f, float h = 0.0f
		);

	~InFlow();
private:
	InFlowData params_;
	InFlowData *gpu_params_; //gpu handle to data

};
*/
#endif