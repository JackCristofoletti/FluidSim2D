#include "fluidsolver.h"

FluidSolver::FluidSolver(int w, int h, double density) : width_(w), height_(h), fluid_density_(density)
{
    cell_size_ = 1.0 / std::min(w, h);

    fluid_concentration_.reset( new FluidQuantity(width_,     height_,     0.5, 0.5, cell_size_) );
    x_velocity_.reset( new FluidQuantity(width_ + 1, height_,     0.0, 0.5, cell_size_) );
    y_velocity_.reset( new FluidQuantity(width_,     height_ + 1, 0.5, 0.0, cell_size_) );

    rhs_pressure_.resize( width_ * height_ , 0 );
    pressure_.resize( width_ * height_ , 0 );

}

FluidSolver::~FluidSolver()
{}

void FluidSolver::Update(double timestep)
{
    BuildRhsPressure();
    Project(600, timestep);
    ApplyPressure(timestep);

    fluid_concentration_->Advect(timestep, *x_velocity_, *y_velocity_);
    x_velocity_->Advect(timestep, *x_velocity_, *y_velocity_);
    y_velocity_->Advect(timestep, *x_velocity_, *y_velocity_);

    /* Make effect of advection visible, since it's not an in-place operation */
    fluid_concentration_->Flip();
    x_velocity_->Flip();
    y_velocity_->Flip();
}

void FluidSolver::AddInflow(double x, double y, double w, double h, double d, double u, double v)
{
    fluid_concentration_->AddInflow(x, y, x + w, y + h, d);
    x_velocity_->AddInflow(x, y, x + w, y + h, u);
    y_velocity_->AddInflow(x, y, x + w, y + h, v);
}

double FluidSolver::CalcMaxTimeStep()
{
    double max_velocity = 0.0;
    for (int y = 0; y < height_; y++)
    {
        for (int x = 0; x < width_; x++)
        {
            /* Average velocity at grid cell center */
            double u = x_velocity_->GridValue(x + 0.5, y + 0.5);
            double v = y_velocity_->GridValue(x + 0.5, y + 0.5);

            double velocity = sqrt(u*u + v*v);
            max_velocity = std::max(max_velocity, velocity);
        }
    }

    /* Fluid should not flow more than two grid cells per iteration */
    double max_timestep = 2.0*cell_size_/max_velocity;

    /* Clamp to sensible maximum value in case of very small velocities */
    return std::min(max_timestep, 1.0);
}

QImage FluidSolver::ToImage()
{
    QByteArray data;
    QImage image(height_, width_, QImage::Format_ARGB32_Premultiplied );
    data.resize( width_ * height_ * 4);


    for (unsigned i=0; i<width_; i++)
    {
        for (unsigned j=0; j<height_; j++)
        {
            int shade = (int)((1.0 - fluid_concentration_->Src()[i + j*width_])*255.0);
                        shade = std::max( std::min(shade, 255), 0 );
            auto value = qRgba(shade, shade, shade, 0xff );
            image.setPixel(i,j,value);
        }
    }
    return image;
}

void FluidSolver::BuildRhsPressure()
{
    double scale = 1.0 / cell_size_;
    for (int y = 0, idx = 0; y < height_; y++)
    {
        for (int x = 0; x < width_; x++, idx++)
        {
            rhs_pressure_[idx] = -scale*(x_velocity_->At(x + 1, y) - x_velocity_->At(x, y) +
                               y_velocity_->At(x, y + 1) - y_velocity_->At(x, y) );
        }
    }
}

void FluidSolver::Project(int limit, double timestep)
{
    double scale = timestep/( fluid_density_* cell_size_ * cell_size_ );

    double maxDelta;
    for (int iter = 0; iter < limit; iter++)
    {
        maxDelta = 0.0;
        for (int y = 0, idx = 0; y < height_; y++)
        {
            for (int x = 0; x < width_; x++, idx++)
            {
                int idx = x + y*width_;

                double diag = 0.0, offDiag = 0.0;

                /* Here we build the matrix implicitly as the five-point
                 * stencil. Grid borders are assumed to be solid, i.e.
                 * there is no fluid outside the simulation domain.
                 */
                if (x > 0)
                {
                    diag    += scale;
                    offDiag -= scale*pressure_[idx - 1];
                }
                if (y > 0)
                {
                    diag    += scale;
                    offDiag -= scale*pressure_[idx - width_];
                }
                if (x < width_ - 1)
                {
                    diag    += scale;
                    offDiag -= scale*pressure_[idx + 1];
                }
                if (y < height_ - 1)
                {
                    diag    += scale;
                    offDiag -= scale*pressure_[idx + width_];
                }

                double newP = (rhs_pressure_[idx] - offDiag)/diag;

                maxDelta = std::max(maxDelta, std::fabs(pressure_[idx] - newP));

                pressure_[idx] = newP;
            }
        }

        if (maxDelta < 1e-5) {
            printf("Exiting solver after %d iterations, maximum change is %f\n", iter, maxDelta);
            return;
        }
    }

    printf("Exceeded budget of %d iterations, maximum change was %f\n", limit, maxDelta);
}

void FluidSolver::ApplyPressure(double timestep)
{
    double scale = timestep/(fluid_density_*cell_size_);
    for (int y = 0, idx = 0; y < height_; y++)
    {
        for (int x = 0; x < width_; x++, idx++)
        {
            x_velocity_->At(x,     y    ) -= scale*pressure_[idx];
            x_velocity_->At(x + 1, y    ) += scale*pressure_[idx];
            y_velocity_->At(x,     y    ) -= scale*pressure_[idx];
            y_velocity_->At(x,     y + 1) += scale*pressure_[idx];
        }
    }

    for (int y = 0; y < height_; y++)
        x_velocity_->At(0, y) = x_velocity_->At(width_, y) = 0.0;
    for (int x = 0; x < width_; x++)
        y_velocity_->At(x, 0) = y_velocity_->At(x, height_) = 0.0;
}
