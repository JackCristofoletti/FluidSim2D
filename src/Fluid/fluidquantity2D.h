#ifndef FLUID_QUANTITY_2D_H
#define FLUID_QUANTITY_2D_H

#include "InFlowData.h"

#include <vector>

#define TX 32 // number of threads per block along x-axis
#define TY 32 // number of threads per block along y-axis
/*
 * A class for representing a fluid quantity (this could be color, velocity, pressure etc. )
 * Note that x,y coordinates refer to x,y in the cartesian sense and not matrix row/column
 */

struct FluidQuantityParams
{
	/* Width and height */
	unsigned width;
	unsigned height;
	/* X and Y offset from top left grid cell.
	* This is (0.5,0.5) for centered quantities such as density,
	* and (0.0, 0.5) or (0.5, 0.0) for jittered quantities like the velocity.
	*/
	double offset_x;
	double offset_y;
	/* Grid cell size */
	double cell_size;
};
class FluidQuantity2D
{
public:
    FluidQuantity2D( unsigned w, unsigned h, double ox, double oy, double cell_size );

    ~FluidQuantity2D();

    //make advection values visible
    void Flip();
    //value buffer
    const double *Src() const;

    /* Advect grid in velocity field u, v with given timestep */
    void Advect(double timestep, FluidQuantity2D &u, FluidQuantity2D &v);

    /* Sets fluid quantity inside the given rect to value `v' */
    void AddInflow(double x0, double y0, double x1, double y1, double v);
	void AddInflow(InFlowData* inflows, unsigned num_inflows, InFlowData::QuantityToAdd quantity);

	void ResetQuantity();

	//gpu value buffers
	double* GetGpuSource();
	double* GetGpuDst() { return dst_gpu_; }
	const FluidQuantityParams* GetGpuFluidParams() const { return gpu_params_; }
	void LoadGpuToCpu();

private:
    /* Memory buffers for fluid quantity */
    std::vector<double> src_;

    /* Width and height */
    int width_;
    int height_;
    /* X and Y offset from top left grid cell.
     * This is (0.5,0.5) for centered quantities such as density,
     * and (0.0, 0.5) or (0.5, 0.0) for jittered quantities like the velocity.
     */
    double offset_x_;
    double offset_y_;
    /* Grid cell size */
    double cell_size_;
	FluidQuantityParams params_;

	//gpu buffers
	FluidQuantityParams* gpu_params_;
	double* src_gpu_;
	double* dst_gpu_;
};

#endif
