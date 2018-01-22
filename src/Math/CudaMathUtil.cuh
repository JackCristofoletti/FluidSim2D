#ifndef CUDA_MATH_UTIL_H
#define CUDA_MATH_UTIL_H

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <host_defines.h>
#include <vector_types.h>

#include <utility>

namespace cuda_math
{
	/* Computes GridSize and BlockSize for width*height elements  using thread sizes of (tx,ty)*/
	std::pair< dim3, dim3 > GetKernelDimensions(unsigned width, unsigned height, unsigned tx, unsigned ty);

	/* Linear interpolate between a and b for x ranging from 0 to 1 */
	template < typename T >
	__host__ __device__ T lerp(T a, T b, T x)
	{
		return (a *(T(1) - x)) + (b * x);
	}

	template < typename T >
	// clamp x to range [min_val, max_val]
	__device__ float clamp(T x, T max_val, T min_val)
	{
		return max( max_val, min(min_val, x) );
	}

	/* Simple forward Euler method for velocity integration in time */
	template < typename T >
	__host__ __device__ void euler2D(T &x_pos, T &y_pos, T timestep, T x_velocity, T y_velocity)
	{
		x_pos -= x_velocity * timestep;
		y_pos -= y_velocity * timestep;
	}

	//return value
	template < typename T >
	__host__ __device__ T At2D(T* buffer, unsigned x, unsigned y, unsigned width, unsigned height)
	{
		return buffer[x + y*width];
	}

	//return reference
	template < typename T >
	__host__ __device__ T& Access2D(T* buffer, unsigned x, unsigned y, unsigned width, unsigned height)
	{
		return buffer[x + y*width];
	}

	// convert floating point rgba color to unsigned integer
	template < typename T >
	__device__ unsigned rgbaToUI(T r, T g, T b, T a)
	{
		r = clamp(r, T(0.0), T(255.0));
		g = clamp(g, T(0.0), T(255.0));
		b = clamp(b, T(0.0), T(255.0));
		a = clamp(a, T(0.0), T(255.0));
		return (unsigned(a) << 24) | (unsigned(b) << 16) | (unsigned(g) << 8) | unsigned(r);
	}

	//initis all data to val
	__global__ void InitGrid(float *grid, unsigned width, unsigned height, double val);
}
#endif