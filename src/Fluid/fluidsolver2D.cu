#include "FluidSolver2D.h"
#include "../Math/CudaMathUtil.cuh"

#include <algorithm>
#include <iostream>

#include <cuda.h>
#include <C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/common/inc/helper_math.h> //vector ops
#include <Windows.h> // need this before interop or there will be compiler errors on windows
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

namespace CudaTextureResources
{
	surface<void, 2> outputSurface;

	// Texture References for 2d fluid sim
	texture<float,  2,  cudaReadModeElementType>   densityTexture;
	texture<float4, 2,  cudaReadModeElementType>   colorTexture;
	texture<float2, 2,  cudaReadModeElementType>   velocityTexture;
	texture<float,  2,  cudaReadModeElementType>   pressureTexture;
	texture<float,  2,  cudaReadModeElementType>   divergenceTexture;
	texture<float4, 2,  cudaReadModeElementType>   obstacleTexture;


	enum FluidQuantities
	{
		  RED
		, GREEN
		, BLUE
		, VEL
		, DENSITY
		, PRESSURE
	};

	template< typename T >
	__global__ void cudaArrayWriteSurface( T* buffer, uint2 gridDim )
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		int index = x + y * gridDim.x;

		if (x < gridDim.x && y < gridDim.y)
		{
			T value = buffer[index];
			surf2Dwrite( value, CudaTextureResources::outputSurface, x * sizeof( T ), y, cudaBoundaryModeClamp );
		}
	}

	//This writes a dev buffer to a texture. Seemingly quicker than copying via cudaMemCpyArray
	template< typename T >
	void WriteArrayToCudaSurface( T* buffer, cudaArray* cudaSurfaceArray, uint2 gridDim)
	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
		cudaError error = cudaBindSurfaceToArray( outputSurface, cudaSurfaceArray, channelDesc );
		if (error != cudaError::cudaSuccess)
		{
			std::cout << cudaGetErrorString( error );
			std::exit( 1 );
		}
		auto thread_dims = cuda_math::GetKernelDimensions( gridDim.x, gridDim.y, TX, TY );
		cudaArrayWriteSurface << <thread_dims.first, thread_dims.second >> > ( buffer, gridDim );
		cudaDeviceSynchronize();
	}
}

//creates rgb texture out of fluid color/density
__global__ void WriteToTexture( unsigned* color_dst, float* density, float4* color, uint2 gridSize )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x + gridSize.y * y;

	if (x < gridSize.x && y < gridSize.y)
	{
		float shade = 255.0f - (1.0 - density[i]) *255.0;
		color_dst[i] = cuda_math::rgbaToUI( color[i].x, color[i].y, color[i].z, shade );
	}
}

__global__ void createObstacleBorder( float4 *obst, uint2 gridSize )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = x + y * gridSize.y;

	if (x< gridSize.x && y< gridSize.y )
	{
		obst[index] = make_float4( 0, 0, 0, 0 );

		if (x == 0) {
			obst[index].w = 1;
		}

		if (x == (gridSize.x - 1)) {
			obst[index].w = 1;
		}

		if ( y == 0) {
			obst[index].w = 1;
		}

		if (y == (gridSize.y - 1)) {
			obst[index].w = 1;
		}
	}
}

__global__ void AddInFlow1DQuantity( float* buffer, uint2 gridSize, float cell_size, InFlowData* inflows, unsigned num_inflows )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = x + y * gridSize.x;
	float2 position = make_float2( float( x ) / float( gridSize.x ), float( y ) / float( gridSize.y ) );

	if (x < gridSize.x && y < gridSize.y)
	{
		for (int i = 0; i < num_inflows; i++)
		{
			//calculate if index is inside of circle
			float2 emiiterPos = make_float2( inflows[i].x, inflows[i].y );
			float2 dist = (position - emiiterPos);

			if (dot( dist, dist ) < (inflows[i].radius*inflows[i].radius))
			{
				if (fabs( buffer[index] ) < fabs( inflows[i].density ))
					buffer[index] = inflows[i].density;
			}
		}
	}
}

__global__ void AddInFlowColor( float4* buffer, uint2 gridSize, float cell_size, InFlowData* inflows, unsigned num_inflows )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = x + y * gridSize.x;
	float2 position = make_float2( float( x ) / float( gridSize.x ), float( y ) / float( gridSize.y ) );
	if (x < gridSize.x && y < gridSize.y)
	{
		for (int i = 0; i < num_inflows; i++)
		{
			//calculate if index is inside of circle
			float2 emiiterPos = make_float2( inflows[i].x, inflows[i].y  );
			float2 dist = (position - emiiterPos);

			if (dot( dist, dist ) <(inflows[i].radius*inflows[i].radius))
			{
				if(fabs( buffer[index].x ) < fabs( inflows[i].red ))
					buffer[index].x = clamp( inflows[i].red + buffer[index].x, 0.0f, 255.0f );
				if (fabs( buffer[index].y ) < fabs( inflows[i].green ))
					buffer[index].y = clamp( inflows[i].green + buffer[index].y, 0.0f, 255.0f );
				if (fabs( buffer[index].z ) < fabs( inflows[i].blue ))
					buffer[index].z = clamp( inflows[i].blue + buffer[index].z, 0.0f, 255.0f );
			}
		}
	}
}

__global__ void AddInFlowVelocityCUDA( float2* velocity, uint2 gridSize, float cell_size, InFlowData* inflows, unsigned num_inflows )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = x + y * gridSize.x;
	float2 position = make_float2( float( x ) / float( gridSize.x ), float( y ) / float( gridSize.y ) );

	if (x < gridSize.x && y < gridSize.y)
	{
		for (int i = 0; i < num_inflows; i++)
		{
			//calculate if index is inside of circle
			float2 emiiterPos = make_float2( inflows[i].x , inflows[i].y );
			float2 dist = (position - emiiterPos);

			if (dot( dist, dist )<(inflows[i].radius*inflows[i].radius))
			{
				float uVel = velocity[index].x;
				float vVel = velocity[index].y;
				
				if (fabs( uVel ) < fabs( inflows[i].u_vel ))
					uVel = inflows[i].u_vel;
				if (fabs( vVel ) < fabs( inflows[i].v_vel ))
					vVel = inflows[i].v_vel;

				velocity[index] = make_float2( uVel, vVel );
			}
		}
	}
}

__global__ void AdvectVel( float2 *vel, float timestep, float dissipation, float2 invGridSize, uint2 gridSize ) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// int index = x + y * blockDim.x * gridDim.x;
	int index = x + y * gridSize.x;

	if (x<gridSize.x && y<gridSize.y) {

		float xc = x + 0.5;
		float yc = y + 0.5;

		float solid = tex2D( CudaTextureResources::obstacleTexture, xc, yc ).w;

		if (solid > 0) {
			vel[index] = make_float2( 0, 0 );
			return;
		}

		// Third Order Runge-Kutta Solver for advection
		float2 firstPos = make_float2( xc, yc );
		float2 firstVel = invGridSize * tex2D( CudaTextureResources::velocityTexture, xc, yc ) * make_float2( (float)gridSize.x, (float)gridSize.y );

		float2 midPos = firstPos - 0.5 * timestep * firstVel;
		float2 midVel = invGridSize * tex2D( CudaTextureResources::velocityTexture, midPos.x, midPos.y )*make_float2( (float)gridSize.x, (float)gridSize.y );

		float2 finalPos = firstPos - 0.75 * timestep * midVel;
		float2 finalVel = invGridSize * tex2D( CudaTextureResources::velocityTexture, finalPos.x, finalPos.y )*make_float2( (float)gridSize.x, (float)gridSize.y );

		float2 pos = firstPos - timestep* (
			firstVel
			+ (3.0 / 9.0) * midVel
			+ (4.0 / 9.0) * finalVel
			);

		vel[index] = (1 - dissipation*timestep) * tex2D( CudaTextureResources::velocityTexture, pos.x, pos.y );
	}

}

__global__ void AdvectColor( float4 *color, float timestep, float dissipation, float2 invGridSize, uint2 gridSize ) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// int index = x + y * blockDim.x * gridDim.x;
	int index = x + y * gridSize.x;

	if (x<gridSize.x && y<gridSize.y) {

		float xc = x + 0.5;
		float yc = y + 0.5;

		float solid = tex2D( CudaTextureResources::obstacleTexture, xc, yc ).w;

		if (solid > 0) {
			color[index] = make_float4( 0.0f );
			return;
		}

		// Third Order Runge-Kutta Solver for advection
		float2 firstPos = make_float2( xc, yc );
		float2 firstVel = invGridSize * tex2D( CudaTextureResources::velocityTexture, xc, yc ) * make_float2( (float)gridSize.x, (float)gridSize.y );
		
		float2 midPos = firstPos - 0.5 * timestep * firstVel;
		float2 midVel = invGridSize * tex2D( CudaTextureResources::velocityTexture, midPos.x, midPos.y )*make_float2( (float)gridSize.x, (float)gridSize.y );
		
		float2 finalPos = firstPos - 0.75 * timestep * midVel;
		float2 finalVel = invGridSize * tex2D( CudaTextureResources::velocityTexture, finalPos.x, finalPos.y )*make_float2( (float)gridSize.x, (float)gridSize.y );
		
		float2 pos = firstPos - timestep* ( 
			  firstVel 
			+ (3.0 / 9.0) * midVel 
			+ (4.0 / 9.0) * finalVel
			);

		color[index] = (1 - dissipation*timestep) * tex2D( CudaTextureResources::colorTexture, pos.x, pos.y );
	}

}

//only density atm
__global__ void Advect1DQuantity( float *dens, float timestep, float dissipation, float2 invGridSize, uint2 gridSize ) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int index = x + y * gridSize.x;

	if (x<gridSize.x && y<gridSize.y) {

		float xc = x + 0.5;
		float yc = y + 0.5;

		float solid = tex2D( CudaTextureResources::obstacleTexture, xc, yc ).w;

		if (solid > 0) {
			dens[index] = 0;
			return;
		}

		// Third Order Runge-Kutta Solver for advection
		float2 firstPos = make_float2( xc, yc );
		float2 firstVel = invGridSize * tex2D( CudaTextureResources::velocityTexture, xc, yc ) * make_float2( (float)gridSize.x, (float)gridSize.y );

		float2 midPos = firstPos - 0.5 * timestep * firstVel;
		float2 midVel = invGridSize * tex2D( CudaTextureResources::velocityTexture, midPos.x, midPos.y )*make_float2( (float)gridSize.x, (float)gridSize.y );

		float2 finalPos = firstPos - 0.75 * timestep * midVel;
		float2 finalVel = invGridSize * tex2D( CudaTextureResources::velocityTexture, finalPos.x, finalPos.y )*make_float2( (float)gridSize.x, (float)gridSize.y );

		float2 pos = firstPos - timestep* (
			firstVel
			+ (3.0 / 9.0) * midVel
			+ (4.0 / 9.0) * finalVel
			);

		dens[index] = (1 - dissipation*timestep) * tex2D( CudaTextureResources::densityTexture, pos.x, pos.y );
	}
}

__global__ void Divergence( float *div, uint2 gridSize, float2 invCellSize ) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// int index = x + y * blockDim.x * gridDim.x;
	int index = x + y * gridSize.x;

	if ( x < gridSize.x && y < gridSize.y ) 
	{
		float xc = x + 0.5;
		float yc = y + 0.5;

		float2 vL = tex2D( CudaTextureResources::velocityTexture, xc - 1, yc );
		float2 vR = tex2D( CudaTextureResources::velocityTexture, xc + 1, yc );
		float2 vT = tex2D( CudaTextureResources::velocityTexture, xc, yc + 1 );
		float2 vB = tex2D( CudaTextureResources::velocityTexture, xc, yc - 1 );

		//obstacles
		float4 oL = tex2D( CudaTextureResources::obstacleTexture, xc - 1, yc );
		float4 oR = tex2D( CudaTextureResources::obstacleTexture, xc + 1, yc );
		float4 oT = tex2D( CudaTextureResources::obstacleTexture, xc, yc + 1 );
		float4 oB = tex2D( CudaTextureResources::obstacleTexture, xc, yc - 1 );

		// Use obstacle velocities for solid cells:
		if (oL.w>0) vL = make_float2( oL.x, oL.y );
		if (oR.w>0) vR = make_float2( oR.x, oR.y );
		if (oT.w>0) vT = make_float2( oT.x, oT.y );
		if (oB.w>0) vB = make_float2( oB.x, oB.y );

		div[index] = 0.5 * (invCellSize.x*(vR.x - vL.x) + invCellSize.y*(vT.y - vB.y));
	}
}

__global__ void JacobiStep( float *pressure, float alpha, float rBeta, uint2 gridSize ) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//int index = x + y * blockDim.x * gridDim.x;
	int index = x + y * gridSize.x;

	if (x<gridSize.x && y<gridSize.y) {

		float xc = x + 0.5;
		float yc = y + 0.5;

		float pL = tex2D( CudaTextureResources::pressureTexture, xc - 1, yc );
		float pR = tex2D( CudaTextureResources::pressureTexture, xc + 1, yc );
		float pT = tex2D( CudaTextureResources::pressureTexture, xc, yc + 1 );
		float pB = tex2D( CudaTextureResources::pressureTexture, xc, yc - 1 );

		float pC = tex2D( CudaTextureResources::pressureTexture, xc, yc );

		//obstacles
		float4 oL = tex2D( CudaTextureResources::obstacleTexture, xc - 1, yc );
		float4 oR = tex2D( CudaTextureResources::obstacleTexture, xc + 1, yc );
		float4 oT = tex2D( CudaTextureResources::obstacleTexture, xc, yc + 1 );
		float4 oB = tex2D( CudaTextureResources::obstacleTexture, xc, yc - 1 );

		// Use center pressure for solid cells (non slip zero boundary):
		if (oL.w>0) pL = pC;
		if (oR.w>0) pR = pC;
		if (oT.w>0) pT = pC;
		if (oB.w>0) pB = pC;

		float dC = tex2D( CudaTextureResources::divergenceTexture, xc, yc );

		pressure[index] = (pL + pR + pB + pT + alpha * dC) * rBeta;
	}
}

//Subtract Pressue gradient from velocity. Insures the divergence free condition
__global__ void Projection( float2 *vel, uint2 gridSize, float2 invCellSize ) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//int index = x + y * blockDim.x * gridDim.x;
	int index = x + y * gridSize.x;

	if (x<gridSize.x && y<gridSize.y) 
	{
		float xc = x + 0.5;
		float yc = y + 0.5;

		float pL = tex2D( CudaTextureResources::pressureTexture, xc - 1, yc );
		float pR = tex2D( CudaTextureResources::pressureTexture, xc + 1, yc );
		float pT = tex2D( CudaTextureResources::pressureTexture, xc, yc + 1 );
		float pB = tex2D( CudaTextureResources::pressureTexture, xc, yc - 1 );

		float pC = tex2D( CudaTextureResources::pressureTexture, xc, yc );

		//obstacles
		float4 oL = tex2D( CudaTextureResources::obstacleTexture, xc - 1, yc );
		float4 oR = tex2D( CudaTextureResources::obstacleTexture, xc + 1, yc );
		float4 oT = tex2D( CudaTextureResources::obstacleTexture, xc, yc + 1 );
		float4 oB = tex2D( CudaTextureResources::obstacleTexture, xc, yc - 1 );

		float2 obstV = make_float2( 0, 0 );
		float2 vMask = make_float2( 1, 1 );

		if (oT.w > 0) { pT = pC; obstV.y = oT.y; vMask.y = 0; }
		if (oB.w > 0) { pB = pC; obstV.y = oB.y; vMask.y = 0; }
		if (oR.w > 0) { pR = pC; obstV.x = oR.x; vMask.x = 0; }
		if (oL.w > 0) { pL = pC; obstV.x = oL.x; vMask.x = 0; }

		float2 grad = 0.5 * make_float2( invCellSize.x*(pR - pL), invCellSize.y*(pT - pB) );

		float2 vNew = tex2D( CudaTextureResources::velocityTexture, xc, yc ) - grad;

		vel[index] = vMask*vNew + obstV;
	}
}

class FluidSolver2DPrivate
{
public:
	FluidSolver2DPrivate( unsigned int w, unsigned int h, double density );
	~FluidSolver2DPrivate();
public:
	/* Fluid quantities */
	std::vector<InFlowData> inflowObjects_;
	uint2		 gridSize;
	float2		 invGridSize; // grid dimensions  
	float2		 invCellSize;
	float cell_size_;
	float fluid_density_;

	//fluid solver buffers
	float2*     dev_vel;
	float*      dev_dens;
	float*      dev_pressure;
	float*      dev_div;
	float4*	    dev_obstacles;

	//other advected quantities
	float4*		dev_color;

	//binded to cuda textures
	cudaArray*  velArray;
	cudaArray*  densArray;
	cudaArray*  pressureArray;
	cudaArray*  divArray;
	cudaArray*  obstArray;
	cudaArray*  colorArray;


	InFlowData* inflowObjectsGpu_;
	bool inflowObjectsDirty_; //flag flowers for reupload
	std::unique_ptr<FluidSolverGLCudaInteropHelper> texture_writer_;
};

class FluidSolverGLCudaInteropHelper
{
public:
	//registers opengl resources with cuda
	FluidSolverGLCudaInteropHelper( unsigned tex_id, unsigned width, unsigned height )
		: tex_id_( tex_id )
	{
		cudaGraphicsGLRegisterImage( &cuda_tex_resource_, tex_id_, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard );

		//set up temp buffer
		unsigned num_texels = width * height;
		unsigned num_values = num_texels * 4;
		std::size_t size_tex_data = sizeof( GLubyte ) * num_values;
		cudaMalloc( (void**)(&cuda_dest_resource_), size_tex_data );
	}

	~FluidSolverGLCudaInteropHelper()
	{
		cudaFree( cuda_dest_resource_ );
	}
	//all pointers are gpu buffers
	void WriteToFluidTexture( uint2 gridDim, float* density, float4* color )
	{
		auto thread_dims = cuda_math::GetKernelDimensions( gridDim.x, gridDim.y, TX, TY );
		WriteToTexture << <thread_dims.first, thread_dims.second >> > (cuda_dest_resource_, density, color, gridDim);
		cudaDeviceSynchronize();
		cudaArray *texture_ptr;

		cudaGraphicsMapResources( 1, &cuda_tex_resource_, 0 );
		cudaGraphicsSubResourceGetMappedArray( &texture_ptr, cuda_tex_resource_, 0, 0 );

		int num_texels = gridDim.x * gridDim.y;
		int num_values = num_texels * 4;
		int size_tex_data = sizeof( GLubyte ) * num_values;
		cudaMemcpyToArray( texture_ptr, 0, 0, cuda_dest_resource_, size_tex_data, cudaMemcpyDeviceToDevice );
		cudaGraphicsUnmapResources( 1, &cuda_tex_resource_, 0 );
	}

private:
	unsigned tex_id_;
	struct cudaGraphicsResource* cuda_tex_resource_;
	unsigned * cuda_dest_resource_;
};

FluidSolver2D::FluidSolver2D( unsigned int w, unsigned int h, double density )
{
	d_.reset( new FluidSolver2DPrivate( w, h, density ) );
	Setup2DTexturesCuda_();
}

FluidSolver2D::~FluidSolver2D()
{
}

void FluidSolver2D::Update( double timestep )
{
	CreateObstacleBorder_();
	//cudaMemcpyToArray( d_->obstArray, 0, 0, d_->dev_obstacles, d_->gridSize.x * d->gridSize.y * sizeof( float4 ), cudaMemcpyDeviceToDevice );
	CudaTextureResources::WriteArrayToCudaSurface( d_->dev_obstacles, d_->obstArray, d_->gridSize );

	AddInFlows_();
	BindTextures_();

	CudaTextureResources::WriteArrayToCudaSurface( d_->dev_vel, d_->velArray, d_->gridSize );
	CudaTextureResources::WriteArrayToCudaSurface( d_->dev_dens, d_->densArray, d_->gridSize );
	CudaTextureResources::WriteArrayToCudaSurface( d_->dev_color, d_->colorArray, d_->gridSize );
	
	AdvectCuda_( timestep );
	
	CudaTextureResources::WriteArrayToCudaSurface( d_->dev_vel, d_->velArray, d_->gridSize );
	CudaTextureResources::WriteArrayToCudaSurface( d_->dev_dens, d_->densArray, d_->gridSize );
	CudaTextureResources::WriteArrayToCudaSurface( d_->dev_color, d_->colorArray, d_->gridSize );

	DivergenceCuda_();
	
	//initial pressure guess
	cudaMemset( d_->dev_pressure, 0, sizeof( float ) *d_->gridSize.x * d_->gridSize.y );
	CudaTextureResources::WriteArrayToCudaSurface( d_->dev_div, d_->divArray, d_->gridSize );

	const int jacIter = 40;
	for (int i = 0; i<jacIter; i++) {
		CudaTextureResources::WriteArrayToCudaSurface( d_->dev_pressure, d_->pressureArray, d_->gridSize );
		JacobiStepCuda_();
	}
	
	CudaTextureResources::WriteArrayToCudaSurface( d_->dev_pressure, d_->pressureArray, d_->gridSize );
	ProjectionCuda_();
	
	UnBindTextures_();
	
	d_->texture_writer_->WriteToFluidTexture( d_->gridSize, d_->dev_dens, d_->dev_color);
}

void FluidSolver2D::AddInflowObject( float d, float u, float v, float r, float g, float b, float x, float y, float rad )
{
	if (d_->inflowObjects_.size() < MAX_NUM_INFLOWS)
	{
		InFlowData new_inflow( u, v, d, r, g, b, x, y, rad );
		d_->inflowObjects_.push_back( new_inflow );
	}
	cudaMemcpy( d_->inflowObjectsGpu_, d_->inflowObjects_.data(), d_->inflowObjects_.size() * sizeof( InFlowData ), cudaMemcpyHostToDevice );
}

const double* FluidSolver2D::ToImage() const
{
	return 0;
}

unsigned FluidSolver2D::GetWidth() const
{
	return d_->gridSize.x;
}

unsigned FluidSolver2D::GetHeight() const
{
	return d_->gridSize.y;
}

void FluidSolver2D::RegisterGLTexture( unsigned tex_id )
{
	d_->texture_writer_.reset( new FluidSolverGLCudaInteropHelper( tex_id, d_->gridSize.x, d_->gridSize.y ) );
}

std::vector<InFlowData>& FluidSolver2D::GetInFlows()
{
	return d_->inflowObjects_;
}

void FluidSolver2D::ReUploadInFlows()
{
	cudaMemcpy( d_->inflowObjectsGpu_, d_->inflowObjects_.data(), d_->inflowObjects_.size() * sizeof( InFlowData ), cudaMemcpyHostToDevice );
}

void FluidSolver2D::ResetSimulation()
{
	//reset texture and arrays to 0
	cudaMemset( d_->dev_pressure, 0, sizeof( float ) * d_->gridSize.x * d_->gridSize.y );
	cudaMemset( d_->dev_color, 0, sizeof( float4 ) * d_->gridSize.x * d_->gridSize.y );
	cudaMemset( d_->dev_vel, 0, sizeof( float2 ) * d_->gridSize.x * d_->gridSize.y );
	cudaMemset( d_->dev_dens, 0, sizeof( float ) * d_->gridSize.x * d_->gridSize.y );
	cudaMemset( d_->dev_pressure, 0, sizeof( float ) * d_->gridSize.x * d_->gridSize.y );
	cudaMemset( d_->dev_obstacles, 0, sizeof( float4 ) * d_->gridSize.x * d_->gridSize.y );
	d_->texture_writer_->WriteToFluidTexture(d_->gridSize, d_->dev_dens, d_->dev_color);

}

void FluidSolver2D::BindTextures_()
{
	cudaChannelFormatDesc descFloat = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc descFloat2 = cudaCreateChannelDesc<float2>();
	cudaChannelFormatDesc descFloat4 = cudaCreateChannelDesc<float4>();

	cudaBindTextureToArray( CudaTextureResources::velocityTexture,   d_->velArray,      descFloat2 );
	cudaBindTextureToArray( CudaTextureResources::densityTexture,    d_->densArray,     descFloat );
	cudaBindTextureToArray( CudaTextureResources::pressureTexture,   d_->pressureArray, descFloat );
	cudaBindTextureToArray( CudaTextureResources::divergenceTexture, d_->divArray,      descFloat );
	cudaBindTextureToArray( CudaTextureResources::colorTexture,		 d_->colorArray,    descFloat4 );
	cudaBindTextureToArray( CudaTextureResources::obstacleTexture,   d_->obstArray,     descFloat4 );
}

void FluidSolver2D::UnBindTextures_()
{
	cudaUnbindTexture( CudaTextureResources::velocityTexture );
	cudaUnbindTexture( CudaTextureResources::densityTexture );
	cudaUnbindTexture( CudaTextureResources::pressureTexture );
	cudaUnbindTexture( CudaTextureResources::divergenceTexture );
	cudaUnbindTexture( CudaTextureResources::colorTexture );
	cudaUnbindTexture( CudaTextureResources::obstacleTexture );
}

void FluidSolver2D::CreateObstacleBorder_()
{
	auto thread_dims = cuda_math::GetKernelDimensions( d_->gridSize.x, d_->gridSize.y, TX, TY );
	createObstacleBorder << <thread_dims.first, thread_dims.second >> >( d_->dev_obstacles, d_->gridSize );
	cudaDeviceSynchronize();
}

void FluidSolver2D::AddInFlows_()
{
	auto thread_dims = cuda_math::GetKernelDimensions( d_->gridSize.x, d_->gridSize.y, TX, TY );

	AddInFlow1DQuantity << <thread_dims.first, thread_dims.second >> > (d_->dev_dens, d_->gridSize, d_->cell_size_, d_->inflowObjectsGpu_, d_->inflowObjects_.size() );
	cudaDeviceSynchronize();
	
	AddInFlowVelocityCUDA << <thread_dims.first, thread_dims.second >> > ( d_->dev_vel, d_->gridSize, d_->cell_size_, d_->inflowObjectsGpu_, d_->inflowObjects_.size() );
	cudaDeviceSynchronize();

	AddInFlowColor << <thread_dims.first, thread_dims.second >> > (d_->dev_color, d_->gridSize, d_->cell_size_, d_->inflowObjectsGpu_, d_->inflowObjects_.size());
	cudaDeviceSynchronize();
}

void FluidSolver2D::AdvectCuda_( float timestep )
{
	auto thread_dims = cuda_math::GetKernelDimensions( d_->gridSize.x, d_->gridSize.y, TX, TY );
	
	AdvectVel << <thread_dims.first, thread_dims.second >> > ( d_->dev_vel, timestep, 0, d_->invGridSize, d_->gridSize );
	cudaDeviceSynchronize();

	Advect1DQuantity << <thread_dims.first, thread_dims.second >> > (d_->dev_dens, timestep, 0, d_->invGridSize, d_->gridSize);
	cudaDeviceSynchronize();

	AdvectColor << <thread_dims.first, thread_dims.second >> > (d_->dev_color, timestep, 0, d_->invGridSize, d_->gridSize);
	cudaDeviceSynchronize();
}

void FluidSolver2D::DivergenceCuda_()
{
	auto thread_dims = cuda_math::GetKernelDimensions( d_->gridSize.x, d_->gridSize.y, TX, TY );
	Divergence<< <thread_dims.first, thread_dims.second >> > (d_->dev_div, d_->gridSize, d_->invCellSize );
	cudaDeviceSynchronize();
}

void FluidSolver2D::JacobiStepCuda_()
{
	float alpha = -(1.0f / d_->invCellSize.x * 1.0f / d_->invCellSize.y);
	float rBeta = 0.25;
	auto thread_dims = cuda_math::GetKernelDimensions( d_->gridSize.x, d_->gridSize.y, TX, TY );
	JacobiStep<< <thread_dims.first, thread_dims.second >> > ( d_->dev_pressure, alpha, rBeta, d_->gridSize );
	cudaDeviceSynchronize();
}

void FluidSolver2D::ProjectionCuda_()
{
	auto thread_dims = cuda_math::GetKernelDimensions( d_->gridSize.x, d_->gridSize.y, TX, TY );
	Projection<< < thread_dims.first, thread_dims.second >> > ( d_->dev_vel, d_->gridSize, d_->invCellSize );
	cudaDeviceSynchronize();
}

void FluidSolver2D::Setup2DTexturesCuda_()
{
	CudaTextureResources::colorTexture.filterMode      = cudaFilterModeLinear;
	CudaTextureResources::densityTexture.filterMode    = cudaFilterModeLinear;
	CudaTextureResources::velocityTexture.filterMode   = cudaFilterModeLinear;
	CudaTextureResources::pressureTexture.filterMode   = cudaFilterModeLinear;
	CudaTextureResources::divergenceTexture.filterMode = cudaFilterModeLinear;
	CudaTextureResources::obstacleTexture.filterMode   = cudaFilterModeLinear;
}

FluidSolver2DPrivate::FluidSolver2DPrivate( unsigned int w, unsigned int h, double density ):
	fluid_density_(density)
{
	gridSize = make_uint2( w, h );
	invGridSize = make_float2( 1.0f / float(w), 1.0f / float(h) );
	invCellSize = make_float2( 1.0f, 1.0f );
	cell_size_ = 1.0 / std::min( w, h );

	cudaChannelFormatDesc descFloat  = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc descFloat2 = cudaCreateChannelDesc<float2>();
	cudaChannelFormatDesc descFloat4 = cudaCreateChannelDesc<float4>();

	cudaMalloc( (void**)&dev_vel, sizeof( float2 ) * w * h  );
	cudaMallocArray( &velArray, &descFloat2, w , h, cudaArraySurfaceLoadStore );

	cudaMalloc( (void**)&dev_dens, sizeof( float ) * w * h );
	cudaMallocArray( &densArray, &descFloat, w , h, cudaArraySurfaceLoadStore );

	cudaMalloc( (void**)&dev_pressure, sizeof( float ) * w * h  );
	cudaMallocArray( &pressureArray, &descFloat, w , h, cudaArraySurfaceLoadStore );

	cudaMalloc( (void**)&dev_div, sizeof( float ) * w * h  );
	cudaMallocArray( &divArray, &descFloat, w , h , cudaArraySurfaceLoadStore );

	cudaMalloc( (void**)&dev_obstacles, sizeof( float4 ) * w * h  );
	cudaMallocArray( &obstArray, &descFloat4, w , h, cudaArraySurfaceLoadStore );

	cudaMalloc( (void**)&dev_color, sizeof( float4 ) * w * h );
	cudaMallocArray( &colorArray, &descFloat4, w, h, cudaArraySurfaceLoadStore );

	cudaMalloc( (void**)&inflowObjectsGpu_, sizeof( InFlowData ) * MAX_NUM_INFLOWS );
}

FluidSolver2DPrivate::~FluidSolver2DPrivate()
{
	cudaFree( dev_vel );
	cudaFreeArray( velArray );
	
	cudaFree( dev_dens );
	cudaFreeArray( densArray );
	
	cudaFree( dev_pressure );
	cudaFreeArray( pressureArray );
	
	cudaFree( dev_div );
	cudaFreeArray( divArray );
	
	cudaFree( dev_obstacles );
	cudaFreeArray( obstArray );

	cudaFree( dev_color );
	cudaFreeArray( colorArray );
}
