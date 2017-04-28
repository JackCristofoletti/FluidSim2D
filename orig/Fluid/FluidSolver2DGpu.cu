#include "FluidSolver2DGpu.h"
#include "../Math/CudaMathUtil.cuh"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <algorithm>

/* Builds the pressure right hand side as the negative divergence */
//NOT THIS ONE
__global__ void BuildRhsPressureGpu(double* rhs_pressure, double cell_size, unsigned my_width, unsigned my_height,
	const double* u, const FluidQuantityParams* u_params, const double* v, const FluidQuantityParams* v_params )
{
	double scale = 1.0 / cell_size;
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned pitch = blockDim.x * gridDim.x;

	for (int i = idx; i < my_width * my_height; i += pitch)
	{
		unsigned iy = i / my_width;
		unsigned ix = i - ( my_width * iy ); // mod(idx,width)
		rhs_pressure[i] = -scale*( At2D( u , ix+1, iy, u_params->width, u_params->height) - At2D(u, ix, iy, u_params->width, u_params->height) +
			At2D(v, ix , iy + 1, v_params->width, v_params->height) - At2D(v ,ix, iy, v_params->width, v_params->height ) );
	}
}

__global__ void ProjectGpu(double* pressure, double* new_pressure, double* rhs_pressure, double* delta_buffer, double timestep, double cell_size, double fluid_density, unsigned width, unsigned height)
{

	double scale = timestep / ( fluid_density* cell_size * cell_size );
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned pitch = blockDim.x * gridDim.x;

	for (int i = idx; i < width * height; i += pitch)
	{
		double diag = 0.0, offDiag = 0.0;

		unsigned iy = i / width;
		unsigned ix = i - ( width * iy ); // mod(idx,pitch)

		/* Here we build the matrix implicitly as the five-point
		* stencil. Grid borders are assumed to be solid, i.e.
		* there is no fluid outside the simulation domain.
		*/
		if (ix > 0)
		{
			diag += scale;
			offDiag -= scale*pressure[idx - 1];
		}
		if (iy > 0)
		{
			diag += scale;
			offDiag -= scale*pressure[idx - width];
		}
		if (ix < width - 1)
		{
			diag += scale;
			offDiag -= scale*pressure[idx + 1];

		}
		if (iy < height - 1)
		{
			diag += scale;
			offDiag -= scale*pressure[idx + width];
		}
		double newP = (rhs_pressure[i] - offDiag) / diag;
		new_pressure[i] = newP;
	}
}

__global__ void ApplyPressureGpu(double timestep, double fluid_density, double cell_size, double* pressure, unsigned my_width, unsigned my_height,
	double* u, const FluidQuantityParams* u_params, double* v, const FluidQuantityParams* v_params)
{
	double scale = timestep / (fluid_density*cell_size);
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned pitch = blockDim.x * gridDim.x;

	for (int i = idx; i < my_width * my_height; i += pitch)
	{
		unsigned iy = i / my_width;
		unsigned ix = i - (my_width * iy); // mod(idx,width)

		Access2D(u, ix, iy, u_params->width, u_params->height) -= scale*pressure[i];
		Access2D(u, ix + 1, iy, u_params->width, u_params->height) += scale*pressure[i];
		
		Access2D(v, ix, iy, v_params->width, v_params->height) -= scale*pressure[i];
		Access2D(v, ix, iy + 1, v_params->width, v_params->height) += scale*pressure[i];
		
	}
}

__global__ void ApplyBoundaryConditions(double timestep, double fluid_density, double cell_size, double* pressure, unsigned my_width, unsigned my_height,
	double* u, const FluidQuantityParams* u_params, double* v, const FluidQuantityParams* v_params)
{
	for (int y = 0; y < my_height; y++)
	{
		Access2D(u, 0, y, u_params->width, u_params->height) = 0.0;
		Access2D(u, my_width, y, u_params->width, u_params->height) = 0.0;
	}
	for (int x = 0; x < my_width; x++)
	{
		Access2D(v, x, 0, v_params->width, v_params->height) = 0.0;
		Access2D(v, x, my_height, v_params->width, v_params->height) = 0.0;
	}
}


enum EDGE { X, Y };


FluidSolver2DGpu::FluidSolver2DGpu(unsigned w, unsigned h, double density)
	: width_( w )
	, height_( h )
	, fluid_density_(density)
{
	cudaMalloc( ( void** ) ( &pressure_ ), w * h * sizeof( double) );
	cudaMalloc( ( void** ) ( &rhs_pressure_ ), w * h * sizeof( double ) );
	cudaMalloc( ( void** )(  &new_pressure_ ), w * h * sizeof( double ) );
	cudaMalloc( ( void** )( &delta_buffer_ ), w * h * sizeof( double ) );

	unsigned blockSize = 256;
	unsigned numBlocks = ((w * h) + blockSize - 1) / (blockSize);
	InitGrid << <numBlocks, blockSize >> >(pressure_, w, h, 0.0);
	InitGrid << <numBlocks, blockSize >> >(rhs_pressure_, w, h, 0.0);
	InitGrid << <numBlocks, blockSize >> >(new_pressure_, w, h, 0.0);
	InitGrid << <numBlocks, blockSize >> >(delta_buffer_, w, h, 0.0);
	cudaDeviceSynchronize();

	cell_size_ = 1.0 / std::min(w, h);
	u_.reset( new FluidQuantity2DGpu( w + 1, h, 0.0, 0.5, cell_size_) );
	v_.reset( new FluidQuantity2DGpu( w, h + 1, 0.5, 0.0, cell_size_) );
	concentration_.reset( new FluidQuantity2DGpu( w, h, 0.5, 0.5, cell_size_ ) );
	cpu_density_.resize( ( w * h ), 0 );
}

void FluidSolver2DGpu::Update(double time_step, unsigned texture_id)
{
	unsigned blockSize = 256;
	unsigned numBlocks = ((width_ * height_) + blockSize - 1) / (blockSize);

	BuildRhsPressureGpu << <numBlocks, blockSize >> > (rhs_pressure_, cell_size_, width_, height_, u_->GetSrcBuffer(), u_->GetParamsGPUBuffer(), v_->GetSrcBuffer(), v_->GetParamsGPUBuffer() );
	cudaDeviceSynchronize();
	Project_(120, time_step);
	ApplyPressureGpu << <numBlocks, blockSize >> >( time_step, fluid_density_, cell_size_, pressure_, width_, height_, u_->GetSrcBuffer(), u_->GetParamsGPUBuffer(), v_->GetSrcBuffer(), v_->GetParamsGPUBuffer() );
	cudaDeviceSynchronize();

	ApplyBoundaryConditions<< <numBlocks, blockSize >> >(time_step, fluid_density_, cell_size_, pressure_, width_, height_, u_->GetSrcBuffer(), u_->GetParamsGPUBuffer(), v_->GetSrcBuffer(), v_->GetParamsGPUBuffer());
	cudaDeviceSynchronize();

	concentration_->Advect(time_step, *u_, *v_);
	u_->Advect(time_step, *u_, *v_);
	v_->Advect(time_step, *u_, *v_);


	/* Make effect of advection visible, since it's not an in-place operation */
	concentration_->Flip();
	u_->Flip();
	v_->Flip();

	cudaMemcpy( cpu_density_.data(), concentration_->Src(), width_ * height_ * sizeof(double), ::cudaMemcpyDeviceToHost );
}

//NOT THIS ONE
void FluidSolver2DGpu::AddInflow(double x, double y, double w, double h, double d, double u, double v)
{
	concentration_->AddInflow(x, y, x + w, y + h, d);
	u_->AddInflow(x, y, x + w, y + h, u);
	v_->AddInflow(x, y, x + w, y + h, v);
}

void FluidSolver2DGpu::BuildRhsPressure_()
{
}

//NOT THIS ONE
void FluidSolver2DGpu::Project_( int limit, double timestep )
{
	unsigned blockSize = 256;
	unsigned numBlocks = ( ( width_ * height_ ) + blockSize - 1) / ( blockSize );
	for (int iter = 0; iter < limit; iter++)
	{
		ProjectGpu << <numBlocks, blockSize >> > ( pressure_, new_pressure_, rhs_pressure_, delta_buffer_, timestep, cell_size_, fluid_density_, width_, height_ );
		cudaDeviceSynchronize();
		std::swap(pressure_, new_pressure_);
	}
}

void FluidSolver2DGpu::ApplyPressure_(double timestep)
{
}
