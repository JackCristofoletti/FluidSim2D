#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

#include "fluidquantity.h"
#include <QImage>
#include <memory>

/* Fluid solver class. Sets up the fluid quantities, forces incompressibility
 * performs advection and adds inflows.
 */
class FluidSolver
{
public:
    FluidSolver(int w, int h, double density);

    ~FluidSolver();

    void Update( double timestep );

    /* Set density and x/y velocity in given rectangle to d/u/v, respectively */
    void AddInflow( double x, double y, double w, double h, double d, double u, double v );

    /* Returns the maximum allowed timestep. Note that the actual timestep
     * taken should usually be much below this to ensure accurate
     * simulation - just never above.
     */
    double CalcMaxTimeStep();

    QImage ToImage();

private:
    /* Builds the pressure right hand side as the negative divergence */
    void BuildRhsPressure();

    /* 
     * The Project operation enforces boundary conditions on solid walls and forces fluid incompressibility ( fluid volume does not change )
     * The operation will run untill either the iteration limit is reached or the error is below a certain threshold
     * Advection algorithms can only be run in divergence free vector fields so this operation must occur first.
     * Implementation can be found on pg 78
     */
    void Project( int limit, double timestep );


    /* Applies the computed pressure to the velocity field */
    void ApplyPressure( double timestep );

private:
    /* Fluid quantities */
    std::unique_ptr<FluidQuantity> fluid_concentration_;
    std::unique_ptr<FluidQuantity> x_velocity_;
    std::unique_ptr<FluidQuantity> y_velocity_;

    /* Width and height */
    int width_;
    int height_;

    /* Grid cell size and fluid density */
    double cell_size_;
    double fluid_density_;

    /* Arrays for: */
    std::vector<double> rhs_pressure_; /* Right hand side of pressure solve */
    std::vector<double> pressure_; /* Pressure solution */
};

#endif;
