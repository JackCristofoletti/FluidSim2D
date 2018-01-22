#ifndef INFLOW_H
#define INFLOW_H

#define MAX_NUM_INFLOWS 10 //max number of flowers in simulation

// Object used to add to fluid quantities
struct InFlowData
{
public:
	InFlowData( float u = 0.0f, float v = 0.0f, float d = 0.0f,
		float r = 0.0f, float g = 0.0f, float b = 0.0f,
		float x0 = 0.0f, float y0 = 0.0f,
		float rad = 0.0f
	);

	//velocity of quantity and concentration
	float u_vel = 0.0f;
	float v_vel = 0.0f;
	float density = 0.0f;
	//0 - 255.0 color values
	float red = 0.0f;
	float blue = 0.0f;
	float green = 0.0f;
	//reigon of influence
	float x = 0.0f, y = 0.0f;
	float radius = 0.0f;
};

#endif