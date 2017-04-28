#include "fluidquantity.h"
#include "../Math/mathutil.h"

#include <algorithm>

FluidQuantity::FluidQuantity( int w, int h, double ox, double oy, double cell_size )
        : width_(w)
        , height_(h)
        , offset_x_(ox)
        , offset_y_(oy)
        , cell_size_(cell_size)
{
    src_.resize( w * h, 0 );
    dst_.resize( w * h, 0 );
}

FluidQuantity::~FluidQuantity()
{}

void FluidQuantity::Flip()
{
    src_.swap( dst_ );
    //swap( src_, dst_ );
}

const double* FluidQuantity::Src() const
{
    return src_.data();
}

/* Read-only and read-write access to grid cells */
double FluidQuantity::At(int x, int y) const
{
    return src_[x + y * width_];
}

double& FluidQuantity::At(int x, int y)
{
    return src_[x + y * width_];
}

double FluidQuantity::GridValue(double x, double y) const
{
    x = std::min( std::max(x - offset_x_, 0.0), width_ - 1.001);
    y = std::min( std::max(y - offset_y_, 0.0), height_ - 1.001);
    int ix = (int)x;
    int iy = (int)y;
    x -= ix;
    y -= iy;

    double x00 = At(ix + 0, iy + 0), x10 = At(ix + 1, iy + 0);
    double x01 = At(ix + 0, iy + 1), x11 = At(ix + 1, iy + 1);

    return lerp(lerp(x00, x10, x), lerp(x01, x11, x), y);
}


void FluidQuantity::Advect(double timestep, const FluidQuantity &u, const FluidQuantity &v)
{
    for (int iy = 0, idx = 0; iy < height_; iy++)
    {
        for (int ix = 0; ix < width_; ix++, idx++)
        {
            double x = ix + offset_x_;
            double y = iy + offset_y_;

            double x_vel = u.GridValue(x, y) / cell_size_;
            double y_vel = v.GridValue(x, y) / cell_size_;

            /* First component: Integrate in time */
            euler2D( x, y, timestep, x_vel, y_vel );

            /* Second component: Interpolate from grid */
            dst_[idx] = GridValue(x, y);
        }
    }
}


void FluidQuantity::AddInflow(double x0, double y0, double x1, double y1, double v)
{
    int ix0 = (int)( ( x0 / cell_size_ ) - offset_x_ );
    int iy0 = (int)( ( y0 / cell_size_ ) - offset_y_ );
    int ix1 = (int)( ( x1 / cell_size_ ) - offset_x_ );
    int iy1 = (int)( ( y1 / cell_size_ ) - offset_y_ );

    for (int y = std::max(iy0, 0); y < std::min(iy1, height_ ); y++)
        for (int x = std::max(ix0, 0); x < std::min(ix1, width_); x++)
            if ( std::fabs( src_[x + y *width_ ] ) < std::fabs(v) )
                src_[x + y * width_] = v;
}
