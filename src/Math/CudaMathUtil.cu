#include "CudaMathUtil.cuh"
namespace cuda_math
{
	std::pair<dim3, dim3> GetKernelDimensions(unsigned width, unsigned height, unsigned tx, unsigned ty)
	{
		const dim3 blockSize(tx, ty);
		const int bx = (width + tx - 1) / tx;
		const int by = (height + ty - 1) / ty;
		const dim3 gridSize
			= dim3(bx, by);
		return std::make_pair(gridSize, blockSize);
	}

	__global__ void InitGrid(float *grid, unsigned width, unsigned height, double val)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		const int i = x + width *y;

		if ((x >= width) || (y >= height))
			return;

		grid[i] = val;
	}
}