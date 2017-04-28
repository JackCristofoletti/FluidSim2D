#include "FluidSolver2D.h"
#include "../Math/CudaMathUtil.cuh"

#include <algorithm>
#include <iostream>

#include <cuda.h>
#include <Windows.h> // need this before interop or there will be compiler errors on windows
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

/* Builds the pressure right hand side as the negative divergence */
__global__ void BuildRhsPressureKernel(double* rhs_pressure, double cell_size, unsigned width, unsigned height,
	const double* u, const FluidQuantityParams* u_params, const double* v, const FluidQuantityParams* v_params)
{
	double scale = 1.0 / cell_size;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x + width *y;

	if ((x >= width) || (y >= height))
		return;

	rhs_pressure[i] = -scale*(cuda_math::At2D(u, x + 1, y, u_params->width, u_params->height) - cuda_math::At2D(u, x, y, u_params->width, u_params->height) +
		cuda_math::At2D(v, x, y + 1, v_params->width, v_params->height) - cuda_math::At2D(v, x, y, v_params->width, v_params->height));
}

__global__ void ProjectionKernel(double* pressure, double* new_pressure, double* rhs_pressure, double timestep, double cell_size, double fluid_density, unsigned width, unsigned height)
{
	double scale = timestep / (fluid_density* cell_size * cell_size);
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x + width *y;

	float diag = 0, offDiag = 0;

	if ((x >= width) || (y >= height))
		return;

	/* Here we build the matrix implicitly as the five-point
	* stencil. Grid borders are assumed to be solid, i.e.
	* there is no fluid outside the simulation domain.
	*/
	if (x > 0)
	{
		diag += scale;
		offDiag -= scale*pressure[i - 1];
	}
	if (y > 0)
	{
		diag += scale;
		offDiag -= scale*pressure[i - width];
	}
	if (x < width - 1)
	{
		diag += scale;
		offDiag -= scale*pressure[i + 1];
	}
	if (y < height - 1)
	{
		diag += scale;
		offDiag -= scale*pressure[i + width];
	}

	double newP = (rhs_pressure[i] - offDiag) / diag;
	new_pressure[i] = newP;
}

__global__ void ApplyPressureKernel(double timestep, double fluid_density, double cell_size, double* pressure, unsigned width, unsigned height,
	double* u, double* dst_u, const FluidQuantityParams* u_params, double* v, double *dst_v, const FluidQuantityParams* v_params)
{
	double scale = timestep / (fluid_density*cell_size);
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float new_u = 0.0;
	float new_v = 0.0;

	if ( (x >= width) || (y >= height) )
		return;

	if ( x > 0 )
		new_u = cuda_math::At2D(u, x, y, u_params->width, u_params->height) - scale*( cuda_math::At2D(pressure, x, y, width, height) - cuda_math::At2D(pressure, x - 1, y, width, height) );
	if ( y > 0 )
		new_v = cuda_math::At2D(v, x, y, v_params->width, v_params->height) - scale*( cuda_math::At2D(pressure, x, y, width, height) - cuda_math::At2D(pressure, x, y-1, width, height ) );

	cuda_math::Access2D(dst_u, x, y, u_params->width, u_params->height) = new_u;
	cuda_math::Access2D(dst_v, x, y, v_params->width, v_params->height) = new_v;
}

__global__ void ApplyBoundaryConditionsUKernel( unsigned width, unsigned height, double* u, const FluidQuantityParams* u_params )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( (y >= height) )
		return;

	cuda_math::Access2D(u, 0, y, u_params->width, u_params->height) = 0.0;
	cuda_math::Access2D(u, width, y, u_params->width, u_params->height) = 0.0;
}

__global__ void ApplyBoundaryConditionsVKernel(unsigned width, unsigned height, double* v, const FluidQuantityParams* v_params )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( ( x >= width ) )
		return;

	cuda_math::Access2D(v, x, 0, v_params->width, v_params->height) = 0.0;
	cuda_math::Access2D(v, x, height, v_params->width, v_params->height) = 0.0;
}

__global__ void WriteToTexture( unsigned* color_dst, unsigned width, unsigned height, double* density, double* red, double* green, double* blue)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x + width *y;

	if ((x >= width) || (y >= height))
		return;

	float shade =  ( 1.0 - cuda_math::At2D(density,x,y,width,height) ) *255.0;
	color_dst[i] = cuda_math::rgbaToUI( static_cast<float> ( red[i] ),  static_cast<float>( green[i] ), static_cast<float>( blue[i] ), shade);
}

class FluidSolverGLCudaInteropHelper
{
public:
	//registers opengl resources with cuda
	FluidSolverGLCudaInteropHelper(unsigned tex_id , unsigned width, unsigned height )
		: tex_id_( tex_id)
	{
		cudaGraphicsGLRegisterImage(&cuda_tex_resource_, tex_id_, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
	
		//set up temp buffer
		unsigned num_texels = width * height;
		unsigned num_values = num_texels * 4;
		std::size_t size_tex_data = sizeof(GLubyte) * num_values;
		cudaMalloc( ( void** )( &cuda_dest_resource_ ), size_tex_data );
	}
	
	~FluidSolverGLCudaInteropHelper()
	{
		cudaFree(cuda_dest_resource_);
	}
	//all pointers are gpu buffers
	void WriteToFluidTexture( unsigned width, unsigned height, double* density, double* red, double* green, double* blue )
	{
		auto thread_dims = cuda_math::GetKernelDimensions(width, height, TX, TY);
		WriteToTexture << <thread_dims.first, thread_dims.second >> > ( cuda_dest_resource_,  width, height, density, red, green, blue );
		cudaDeviceSynchronize();
		cudaArray *texture_ptr;

		cudaGraphicsMapResources(1, &cuda_tex_resource_, 0);
		cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource_, 0, 0);
		
		int num_texels = width * height;
		int num_values = num_texels * 4;
		int size_tex_data = sizeof(GLubyte) * num_values;
		cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource_, size_tex_data, cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &cuda_tex_resource_, 0);
	}

private:
	unsigned tex_id_;
	struct cudaGraphicsResource* cuda_tex_resource_;
	unsigned * cuda_dest_resource_;
};

FluidSolver2D::FluidSolver2D(unsigned int w, unsigned int h, double density) : width_(w), height_(h), fluid_density_(density)
{
	cell_size_ = 1.0 / std::min(w, h);

	fluid_concentration_.reset(new FluidQuantity2D(width_, height_, 0.5, 0.5, cell_size_));
	u_velocity_.reset(new FluidQuantity2D(width_ + 1, height_, 0.0, 0.5, cell_size_));
	v_velocity_.reset(new FluidQuantity2D(width_, height_ + 1, 0.5, 0.0, cell_size_));

	red_.reset(new FluidQuantity2D(width_, height_, 0.5, 0.5, cell_size_));
	green_.reset(new FluidQuantity2D(width_, height_, 0.5, 0.5, cell_size_));
	blue_.reset(new FluidQuantity2D(width_, height_, 0.5, 0.5, cell_size_));

	//gpu buffer allocation
	cudaMalloc((void**)(&rhs_pressure_gpu_), w * h * sizeof(double));
	cudaMalloc((void**)(&pressure_gpu_), w * h * sizeof(double));
	cudaMalloc((void**)(&projected_pressure_gpu_), w * h * sizeof(double));
	cudaMalloc((void**)(&inflow_objects_gpu_), MAX_NUM_INFLOWS * sizeof(InFlowData));

	auto thread_dims = cuda_math::GetKernelDimensions(width_, height_, TX, TY);
	
	cuda_math::InitGrid<<<thread_dims.first,thread_dims.second>>>(pressure_gpu_, w, h, 0.0);
	cudaDeviceSynchronize();
}

FluidSolver2D::~FluidSolver2D()
{
	cudaFree(rhs_pressure_gpu_);
	cudaFree(pressure_gpu_);
	cudaFree(projected_pressure_gpu_);
	cudaFree(inflow_objects_gpu_);
}

void FluidSolver2D::Update(double timestep)
{
	fluid_concentration_->AddInflow(inflow_objects_gpu_, inflow_objects_.size(), InFlowData::QuantityToAdd::DENSITY);
	u_velocity_->AddInflow(inflow_objects_gpu_, inflow_objects_.size(), InFlowData::QuantityToAdd::U_VEL);
	v_velocity_->AddInflow(inflow_objects_gpu_, inflow_objects_.size(), InFlowData::QuantityToAdd::V_VEL);

	red_->AddInflow(inflow_objects_gpu_, inflow_objects_.size(), InFlowData::QuantityToAdd::RED);
	green_->AddInflow(inflow_objects_gpu_, inflow_objects_.size(), InFlowData::QuantityToAdd::GREEN);
	blue_->AddInflow(inflow_objects_gpu_, inflow_objects_.size(), InFlowData::QuantityToAdd::BLUE);

	BuildRhsPressure_();
	Project_(600, timestep);
	ApplyPressure_(timestep);

	fluid_concentration_->Advect(timestep, *u_velocity_, *v_velocity_);
	red_->Advect(timestep, *u_velocity_, *v_velocity_);
	green_->Advect(timestep, *u_velocity_, *v_velocity_);
	blue_->Advect(timestep, *u_velocity_, *v_velocity_);
	u_velocity_->Advect(timestep, *u_velocity_, *v_velocity_);
	v_velocity_->Advect(timestep, *u_velocity_, *v_velocity_);

	/* Make effect of advection visible, since it's not an in-place operation  */
	fluid_concentration_->Flip();
	u_velocity_->Flip();
	v_velocity_->Flip();
	red_->Flip();
	green_->Flip();
	blue_->Flip();

	//render to a color texture previously registered with the solver.
	texture_writer_->WriteToFluidTexture(width_, height_, fluid_concentration_->GetGpuSource(), red_->GetGpuSource(), green_->GetGpuSource(), blue_->GetGpuSource() );
}

void FluidSolver2D::AddInflow(double x, double y, double w, double h, double d, double u, double v)
{
	fluid_concentration_->AddInflow(x, y, x + w, y + h, d);
	u_velocity_->AddInflow(x, y, x + w, y + h, u);
	v_velocity_->AddInflow(x, y, x + w, y + h, v);
}

void FluidSolver2D::AddInflowObject(double d, double u, double v, double r, double g, double b, double x, double y, double w, double h)
{
	if ( inflow_objects_.size() < MAX_NUM_INFLOWS )
	{
		InFlowData new_inflow(u, v, d, r, g, b, x, y, w, h);
		inflow_objects_.push_back(new_inflow);
	}
	cudaMemcpy(inflow_objects_gpu_, inflow_objects_.data(), inflow_objects_.size() * sizeof(InFlowData), cudaMemcpyHostToDevice);
}

double FluidSolver2D::CalcMaxTimeStep()
{
	/*
	double max_velocity = 0.0;
	for (int y = 0; y < height_; y++)
	{
		for (int x = 0; x < width_; x++)
		{
			/* Average velocity at grid cell center 
			double u = u_velocity_->GridValue(x + 0.5, y + 0.5);
			double v = v_velocity_->GridValue(x + 0.5, y + 0.5);

			double velocity = sqrt(u*u + v*v);
			max_velocity = std::max(max_velocity, velocity);
		}
	}

	/* Fluid should not flow more than two grid cells per iteration 
	double max_timestep = 2.0*cell_size_ / max_velocity;

	/* Clamp to sensible maximum value in case of very small velocities 
	return std::min(max_timestep, 1.0);
	*/
	return 0.0;
}

const double* FluidSolver2D::ToImage() const
{
	return fluid_concentration_->Src();
}

void FluidSolver2D::RegisterGLTexture(unsigned tex_id)
{
	texture_writer_.reset(new FluidSolverGLCudaInteropHelper( tex_id, width_, height_ ) );
}

void FluidSolver2D::ReUploadInFlows()
{
	cudaMemcpy(inflow_objects_gpu_, inflow_objects_.data(), inflow_objects_.size() * sizeof(InFlowData), cudaMemcpyHostToDevice);
}

void FluidSolver2D::ResetSimulation()
{
	auto thread_dims = cuda_math::GetKernelDimensions(width_, height_, TX, TY);
	cuda_math::InitGrid << <thread_dims.first, thread_dims.second >> >(pressure_gpu_, width_, height_, 0.0);

	fluid_concentration_->ResetQuantity();
	u_velocity_->ResetQuantity();
	v_velocity_->ResetQuantity();
	red_->ResetQuantity();
	green_->ResetQuantity();
	blue_->ResetQuantity();

	//reset texture to 0
	texture_writer_->WriteToFluidTexture(width_, height_, fluid_concentration_->GetGpuSource(), red_->GetGpuSource(), green_->GetGpuSource(), blue_->GetGpuSource());
}

void FluidSolver2D::BuildRhsPressure_()
{
	auto thread_dims = cuda_math::GetKernelDimensions(width_, height_, TX, TY);

	BuildRhsPressureKernel<< <thread_dims.first, thread_dims.second >> > (rhs_pressure_gpu_, cell_size_, width_, height_, u_velocity_->GetGpuSource(), u_velocity_->GetGpuFluidParams(),
		v_velocity_->GetGpuSource(), v_velocity_->GetGpuFluidParams());
	
	cudaDeviceSynchronize();
}

void FluidSolver2D::Project_(int limit, double timestep)
{
	auto thread_dims = cuda_math::GetKernelDimensions(width_, height_, TX, TY);

	for (int iter = 0; iter < 40; iter++)
	{
		ProjectionKernel << < thread_dims.first, thread_dims.second >> > (pressure_gpu_, projected_pressure_gpu_, rhs_pressure_gpu_, timestep, cell_size_, fluid_density_, width_, height_);
		cudaDeviceSynchronize();
		std::swap(pressure_gpu_, projected_pressure_gpu_);
	}
}

void FluidSolver2D::ApplyPressure_(double timestep)
{
	auto thread_dims = cuda_math::GetKernelDimensions(width_, height_, TX, TY);
	
	ApplyPressureKernel << < thread_dims.first, thread_dims.second >> > (timestep, fluid_density_, cell_size_, pressure_gpu_, width_, height_,
		u_velocity_->GetGpuSource(), u_velocity_->GetGpuDst(), u_velocity_->GetGpuFluidParams()
		, v_velocity_->GetGpuSource(), v_velocity_->GetGpuDst(), v_velocity_->GetGpuFluidParams());
	cudaDeviceSynchronize();

	ApplyBoundaryConditionsUKernel << < thread_dims.first, thread_dims.second >> >(width_, height_, u_velocity_->GetGpuDst(), u_velocity_->GetGpuFluidParams());
	ApplyBoundaryConditionsVKernel << < thread_dims.first, thread_dims.second >> > (width_, height_, v_velocity_->GetGpuDst(), v_velocity_->GetGpuFluidParams());
	cudaDeviceSynchronize();

	//write back to src buffers
	u_velocity_->Flip();
	v_velocity_->Flip();
}

