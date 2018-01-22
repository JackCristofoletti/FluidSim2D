#include "InFlowData.h"

InFlowData::InFlowData(float u, float v, float d, float r, float g, float b, float x0, float y0, float rad)
	: u_vel(u)
	, v_vel(v)
	, density(d)
	, red(r)
	, green(g)
	, blue(b)
	, x(x0)
	, y(y0)
	, radius(rad)
{}
