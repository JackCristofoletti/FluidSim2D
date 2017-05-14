#include "FluidQuantity2D.h"
#include "../Math/CudaMathUtil.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <algorithm>

/* Sets fluid quantity inside { {x0,yo} , {x1,y1} } rect to value `v' */
__global__ void AddInFlowKernel(double* src, double x0, double y0, double x1, double y1, double v, FluidQuantityParams* my_params)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x + my_params->width *y; 

	if ( ( x >= my_params->width ) || ( y >= my_params->height ) ) 
		return;

	int ix0 = (int)((x0 / my_params->cell_size) - my_params->offset_x);
	int iy0 = (int)((y0 / my_params->cell_size) - my_params->offset_y);
	int ix1 = (int)((x1 / my_params->cell_size) - my_params->offset_x);
	int iy1 = (int)((y1 / my_params->cell_size) - my_params->offset_y);

	if ( (x <= ix1 && x >= ix0) && (y <= iy1 && y >= iy0) && fabs(src[i]) < fabs(v) )
		src[i] = v;
}

__global__ void AddInFlowKernel(double* src, FluidQuantityParams* my_params, InFlowData* inflows, unsigned num_inflows, InFlowData::QuantityToAdd quantity )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = x + my_params->width *y;

	if ((x >= my_params->width) || (y >= my_params->height))
		return;

	for ( int i = 0; i < num_inflows; i++ )
	{
		//calculate if index is inside rectangle of influence
		int ix0 = (int)( ( inflows[i].x / my_params->cell_size ) - my_params->offset_x );
		int iy0 = (int)( ( inflows[i].y / my_params->cell_size ) - my_params->offset_y );

		int ix1 = (int)( ( ( inflows[i].x + inflows[i].width ) / my_params->cell_size ) - my_params->offset_x );
		int iy1 = (int)( ( ( inflows[i].y + inflows[i].height ) / my_params->cell_size) - my_params->offset_y );

		if ((x <= ix1 && x >= ix0) && (y <= iy1 && y >= iy0))
		{
			double value;
			switch (quantity)
			{
				case InFlowData::QuantityToAdd::U_VEL: { value = inflows[i].u_vel; break; }
				case InFlowData::QuantityToAdd::V_VEL: { value = inflows[i].v_vel; break; }
				case InFlowData::QuantityToAdd::DENSITY: { value = inflows[i].density; break; }
				case InFlowData::QuantityToAdd::RED: { value = inflows[i].red; break; }
				case InFlowData::QuantityToAdd::GREEN: { value = inflows[i].green; break; }
				case InFlowData::QuantityToAdd::BLUE: { value = inflows[i].blue; break; }
				default: break;
			}

			if (fabs(src[index]) < fabs(value))
				src[index] = value;
		}

	}
}

__global__ void AdvectKernel(double timestep, double* u, const FluidQuantityParams* u_params,
	double* v, const FluidQuantityParams* v_params, double* src, double* dst, const FluidQuantityParams* my_params)
{
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = ix + my_params->width * iy;

	if ((ix >= my_params->width) || (iy >= my_params->height))
		return;

	double x = ix + my_params->offset_x;
	double y = iy + my_params->offset_y;

	double x_vel = cuda_math::GridValue2D(u, x, y, u_params->offset_x, u_params->offset_y, u_params->width, u_params->height) / my_params->cell_size;
	double y_vel = cuda_math::GridValue2D(v, x, y, v_params->offset_x, v_params->offset_y, v_params->width, v_params->height) / my_params->cell_size;

	/* Integrate in time 3rd order Runge-Kutta Solver*/
	{
		double firstU = cuda_math::GridValue2D(u, x, y, u_params->offset_x, u_params->offset_y, u_params->width, u_params->height) / my_params->cell_size;
		double firstV = cuda_math::GridValue2D(v, x, y, v_params->offset_x, v_params->offset_y, v_params->width, v_params->height) / my_params->cell_size;

		double midX = x - 0.5 * timestep * firstU;
		double midY = y - 0.5 * timestep * firstV;

		double midU = cuda_math::GridValue2D(u, midX, midY, u_params->offset_x, u_params->offset_y, u_params->width, u_params->height) / my_params->cell_size;
		double midV = cuda_math::GridValue2D(v, midX, midY, v_params->offset_x, v_params->offset_y, v_params->width, v_params->height) / my_params->cell_size;

		double lastX = x - 0.75 * timestep * midU;
		double lastY = y - 0.75 * timestep * midV;

		double lastU = cuda_math::GridValue2D(u, lastX, lastY, u_params->offset_x, u_params->offset_y, u_params->width, u_params->height) / my_params->cell_size;
		double lastV = cuda_math::GridValue2D(v, lastX, lastY, v_params->offset_x, v_params->offset_y, v_params->width, v_params->height) / my_params->cell_size;

		x -= timestep*(firstU + (3.0 / 9.0) * midU + (4.0 / 9.0) * lastU);
		y -= timestep*(firstV + (3.0 / 9.0) * midV + (4.0 / 9.0) * lastV);
	}

	dst[idx] = cuda_math::GridValue2D(src, x, y, my_params->offset_x, my_params->offset_y, my_params->width, my_params->height);
}

FluidQuantity2D::FluidQuantity2D(unsigned int w, unsigned int h, double ox, double oy, double cell_size)
	: width_(w)
	, height_(h)
	, offset_x_(ox)
	, offset_y_(oy)
	, cell_size_(cell_size)
{
	params_ = { w , h, ox , oy, cell_size };
	src_.resize(w * h, 0);
	
	auto thread_dims = cuda_math::GetKernelDimensions(width_, height_, TX, TY);

	//gpu allocation
	cudaMalloc((void**)(&gpu_params_), sizeof(FluidQuantityParams));
	cudaMalloc((void**)(&src_gpu_), w * h * sizeof(double));
	cudaMalloc((void**)(&dst_gpu_), w * h * sizeof(double));
	cudaMemcpy(gpu_params_, &params_, sizeof(FluidQuantityParams), cudaMemcpyHostToDevice);
	cuda_math::InitGrid << <thread_dims.first, thread_dims.second >> >( src_gpu_, w, h, 0 );
	cudaDeviceSynchronize();
}

FluidQuantity2D::~FluidQuantity2D()
{
	//destroy gpu resources
	cudaFree(gpu_params_);
	cudaFree(src_gpu_);
	cudaFree(dst_gpu_);
}

void FluidQuantity2D::Flip()
{
	std::swap(src_gpu_, dst_gpu_);
}

const double* FluidQuantity2D::Src() const
{
	return src_.data();
}

void FluidQuantity2D::Advect(double timestep, FluidQuantity2D &u, FluidQuantity2D &v)
{
	auto thread_dims = cuda_math::GetKernelDimensions(width_, height_, TX, TY);
	AdvectKernel << <thread_dims.first, thread_dims.second >> > (timestep, u.GetGpuSource(), u.GetGpuFluidParams(), v.GetGpuSource(), v.GetGpuFluidParams(), src_gpu_, dst_gpu_, gpu_params_);
	cudaDeviceSynchronize();
}

void FluidQuantity2D::AddInflow(double x0, double y0, double x1, double y1, double v)
{
	auto thread_dims = cuda_math::GetKernelDimensions(width_, height_, TX, TY);
	AddInFlowKernel << <thread_dims.first, thread_dims.second >> > (src_gpu_, x0, y0, x1, y1, v, gpu_params_);
	cudaDeviceSynchronize();
}

void FluidQuantity2D::AddInflow(InFlowData * inflows, unsigned num_inflows, InFlowData::QuantityToAdd quantity)
{
	auto thread_dims = cuda_math::GetKernelDimensions(width_, height_, TX, TY);
	AddInFlowKernel << <thread_dims.first, thread_dims.second >> > (src_gpu_, gpu_params_, inflows, num_inflows, quantity);
	cudaDeviceSynchronize();
}

void FluidQuantity2D::ResetQuantity()
{
	auto thread_dims = cuda_math::GetKernelDimensions(width_, height_, TX, TY);
	cuda_math::InitGrid << <thread_dims.first, thread_dims.second >> >( src_gpu_, width_, height_, 0.0);
}

double * FluidQuantity2D::GetGpuSource()
{
	return src_gpu_;
}

void FluidQuantity2D::LoadGpuToCpu()
{
	cudaMemcpy(src_.data(), src_gpu_, width_ * height_ * sizeof(double), ::cudaMemcpyDeviceToHost);
}
