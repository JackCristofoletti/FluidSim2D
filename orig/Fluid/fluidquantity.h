#ifndef FLUID_QUANTITY_H
#define FLUID_QUANTITY_H

#include <vector>

/*
 * A class for representing a fluid quantity (this could be color, velocity, pressure etc. )
 * Note that x,y coordinates refer to x,y in the cartesian sense and not matrix row/column
 */
class FluidQuantity
{
public:
    FluidQuantity( int w, int h, double ox, double oy, double cell_size );

    ~FluidQuantity();

    //make advection values visible
    void Flip();
    //value buffer
    const double *Src() const;

    /* Read-only and read-write access to grid cells */
    double At(int x, int y) const;
    double &At(int x, int y);

    /* Linear intERPolate on grid at coordinates (x, y).
     * Coordinates will be clamped to lie in simulation domain
     */
    double GridValue(double x, double y) const;

    /* Advect grid in velocity field u, v with given timestep */
    void Advect(double timestep, const FluidQuantity &u, const FluidQuantity &v);

    /* Sets fluid quantity inside the given rect to value `v' */
    void AddInflow(double x0, double y0, double x1, double y1, double v);

private:
    /* Memory buffers for fluid quantity */
    std::vector<double> src_;
    std::vector<double> dst_;

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

};

#endif
