#include "../Math/CudaMathUtil.cuh"
#include "FluidQuantity2DGpu.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <algorithm>

__global__ void AdvectGPU( double timestep, const double* u,  const FluidQuantityParams* u_params,
	const double* v, const FluidQuantityParams* v_params, double* src, double* dst, const FluidQuantityParams* my_params )
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned pitch = blockDim.x * gridDim.x;

	for (int i = idx; i < my_params->width * my_params->height; i += pitch )
	{
		unsigned iy = i / my_params->width;
		unsigned ix = i - (my_params->width * iy ); // mod(idx,width)

		double x = ix + my_params->offset_x;
		double y = iy + my_params->offset_y;

		double x_vel = GridValue2D(u, x, y, u_params->offset_x, u_params->offset_y, u_params->width, u_params->height ) / my_params->cell_size;
		double y_vel = GridValue2D(v, x, y, v_params->offset_x, v_params->offset_y, v_params->width, v_params->height) / my_params->cell_size;

		/* First component: Integrate in time */
		euler2D(x, y, timestep, x_vel, y_vel);

		dst[idx] = GridValue2D(src, x, y, my_params->offset_x, my_params->offset_y, my_params->width, my_params->height);
	}
}

/* Sets fluid quantity inside { {x0,yo} , {x1,y1} } rect to value `v' */
__global__ void AddInFlowGPU( double* src, double x0, double y0, double x1, double y1, double v, FluidQuantityParams* my_params )
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned pitch = blockDim.x * gridDim.x;

	int ix0 = (int)((x0 / my_params->cell_size) - my_params->offset_x);
	int iy0 = (int)((y0 / my_params->cell_size) - my_params->offset_x);
	int ix1 = (int)((x1 / my_params->cell_size) - my_params->offset_x);
	int iy1 = (int)((y1 / my_params->cell_size) - my_params->offset_x);

	for (int i = idx; i < my_params->width * my_params->height; i += pitch)
	{
		unsigned iy = i / my_params->width;
		unsigned ix = i - ( my_params->width * iy ); // mod(idx,width)
		
		if ( ( ix <= ix1 && ix >= ix0 ) && ( iy <= iy1 && iy >= iy0 ) )
			src[idx] = v;
	}
}

FluidQuantity2DGpu::FluidQuantity2DGpu(unsigned w, unsigned h, double ox, double oy, double cell_size)
{
	params_ = { w , h, ox , oy, cell_size };

	cudaMalloc( ( void** ) ( &src_ ), w * h * sizeof( double ) );
	cudaMalloc( ( void** ) ( &dst_ ), w * h * sizeof( double ) );

	cudaMalloc( ( void** )( &gpu_params_ ), sizeof( FluidQuantityParams) );

	unsigned blockSize = 256;
	unsigned numBlocks = ( ( w * h )  + blockSize - 1) / ( blockSize );
	InitGrid << <numBlocks, blockSize >> >( src_, w, h, 0.0 );
	InitGrid << <numBlocks, blockSize >> >( dst_, w, h, 0.0 );
	cudaDeviceSynchronize();
	cudaMemcpy( gpu_params_, &params_, sizeof( FluidQuantityParams ), cudaMemcpyHostToDevice );
}

FluidQuantity2DGpu::FluidQuantity2DGpu(const FluidQuantityParams & params)
{
}

FluidQuantity2DGpu::~FluidQuantity2DGpu()
{
}

void FluidQuantity2DGpu::Flip()
{
	std::swap(dst_, src_);
}

const double * FluidQuantity2DGpu::Src() const
{
	return src_;
}
void FluidQuantity2DGpu::Advect(double timestep, const FluidQuantity2DGpu & u, const FluidQuantity2DGpu & v)
{
	unsigned blockSize = 256;
	unsigned numBlocks = ( ( params_.width * params_.height ) + blockSize - 1 ) / ( blockSize );
	AdvectGPU<< <numBlocks, blockSize >> >( timestep, u.GetSrcBuffer(), u.GetParamsGPUBuffer(), v.GetSrcBuffer(), v.GetParamsGPUBuffer(), src_, dst_, gpu_params_ );
	cudaDeviceSynchronize();
}

void FluidQuantity2DGpu::AddInflow(double x0, double y0, double x1, double y1, double v)
{
	unsigned blockSize = 256;
	unsigned numBlocks = ( (params_.width * params_.height ) + blockSize - 1 ) / (blockSize);
	AddInFlowGPU<< <numBlocks, blockSize >> >( src_, x0, y0, x1, y1, v, gpu_params_);
	cudaDeviceSynchronize();
}
