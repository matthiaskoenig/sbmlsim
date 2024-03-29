#ifndef _amici_TPL_MODELNAME_h
#define _amici_TPL_MODELNAME_h
#include <cmath>
#include <memory>
#include <gsl/gsl-lite.hpp>

#include "amici/model_ode.h"

namespace amici {

class Solver;

namespace model_icg_body {

extern std::array<const char*, 35> parameterNames;
extern std::array<const char*, 0> fixedParameterNames;
extern std::array<const char*, 14> stateNames;
extern std::array<const char*, 70> observableNames;
extern std::array<const ObservableScaling, 70> observableScalings;
extern std::array<const char*, 72> expressionNames;
extern std::array<const char*, 35> parameterIds;
extern std::array<const char*, 0> fixedParameterIds;
extern std::array<const char*, 14> stateIds;
extern std::array<const char*, 70> observableIds;
extern std::array<const char*, 72> expressionIds;
extern std::array<int, 13> stateIdxsSolver;
extern std::array<bool, 0> rootInitialValues;

extern void Jy_icg_body(realtype *Jy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my);
extern void dJydsigma_icg_body(realtype *dJydsigma, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my);
extern void dJydy_icg_body(realtype *dJydy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my);
extern void dJydy_colptrs_icg_body(SUNMatrixWrapper &colptrs, int index);
extern void dJydy_rowvals_icg_body(SUNMatrixWrapper &rowvals, int index);







extern void dwdp_icg_body(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *dtcldp);
extern void dwdp_colptrs_icg_body(SUNMatrixWrapper &colptrs);
extern void dwdp_rowvals_icg_body(SUNMatrixWrapper &rowvals);
extern void dwdx_icg_body(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl);
extern void dwdx_colptrs_icg_body(SUNMatrixWrapper &colptrs);
extern void dwdx_rowvals_icg_body(SUNMatrixWrapper &rowvals);
extern void dwdw_icg_body(realtype *dwdw, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl);
extern void dwdw_colptrs_icg_body(SUNMatrixWrapper &colptrs);
extern void dwdw_rowvals_icg_body(SUNMatrixWrapper &rowvals);
extern void dxdotdw_icg_body(realtype *dxdotdw, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void dxdotdw_colptrs_icg_body(SUNMatrixWrapper &colptrs);
extern void dxdotdw_rowvals_icg_body(SUNMatrixWrapper &rowvals);
extern void dxdotdp_explicit_icg_body(realtype *dxdotdp_explicit, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void dxdotdp_explicit_colptrs_icg_body(SUNMatrixWrapper &colptrs);
extern void dxdotdp_explicit_rowvals_icg_body(SUNMatrixWrapper &rowvals);
extern void dxdotdx_explicit_icg_body(realtype *dxdotdx_explicit, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void dxdotdx_explicit_colptrs_icg_body(SUNMatrixWrapper &colptrs);
extern void dxdotdx_explicit_rowvals_icg_body(SUNMatrixWrapper &rowvals);
extern void dydx_icg_body(realtype *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void dydp_icg_body(realtype *dydp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *tcl, const realtype *dtcldp);




extern void sigmay_icg_body(realtype *sigmay, const realtype t, const realtype *p, const realtype *k, const realtype *y);




extern void w_icg_body(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl);
extern void x0_icg_body(realtype *x0, const realtype t, const realtype *p, const realtype *k);



extern void xdot_icg_body(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void y_icg_body(realtype *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);





extern void x_rdata_icg_body(realtype *x_rdata, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k);
extern void x_solver_icg_body(realtype *x_solver, const realtype *x_rdata);
extern void total_cl_icg_body(realtype *total_cl, const realtype *x_rdata, const realtype *p, const realtype *k);
extern void dx_rdatadx_solver_icg_body(realtype *dx_rdatadx_solver, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k);
extern void dx_rdatadx_solver_colptrs_icg_body(SUNMatrixWrapper &colptrs);
extern void dx_rdatadx_solver_rowvals_icg_body(SUNMatrixWrapper &rowvals);

extern void dx_rdatadtcl_icg_body(realtype *dx_rdatadtcl, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k);
extern void dx_rdatadtcl_colptrs_icg_body(SUNMatrixWrapper &colptrs);
extern void dx_rdatadtcl_rowvals_icg_body(SUNMatrixWrapper &rowvals);

extern void dtotal_cldx_rdata_icg_body(realtype *dtotal_cldx_rdata, const realtype *x_rdata, const realtype *p, const realtype *k, const realtype *tcl);
extern void dtotal_cldx_rdata_colptrs_icg_body(SUNMatrixWrapper &colptrs);
extern void dtotal_cldx_rdata_rowvals_icg_body(SUNMatrixWrapper &rowvals);
/**
 * @brief AMICI-generated model subclass.
 */
class Model_icg_body : public amici::Model_ODE {
  public:
    /**
     * @brief Default constructor.
     */
    Model_icg_body()
        : amici::Model_ODE(
              amici::ModelDimensions(
                  14,                            // nx_rdata
                  14,                        // nxtrue_rdata
                  13,                           // nx_solver
                  13,                       // nxtrue_solver
                  0,                    // nx_solver_reinit
                  35,                                  // np
                  0,                                  // nk
                  70,                                  // ny
                  70,                              // nytrue
                  0,                                  // nz
                  0,                              // nztrue
                  0,                              // nevent
                  1,                          // nobjective
                  72,                                  // nw
                  27,                               // ndwdx
                  196,                               // ndwdp
                  56,                               // ndwdw
                  32,                            // ndxdotdw
                  std::vector<int>{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},                              // ndjydy
                  13,                    // ndxrdatadxsolver
                  1,                        // ndxrdatadtcl
                  1,                        // ndtotal_cldx_rdata
                  0,                                       // nnz
                  13,                                 // ubw
                  13                                  // lbw
              ),
              amici::SimulationParameters(
                  std::vector<realtype>{}, // fixedParameters
                  std::vector<realtype>{1000.0, 0.001, 75.0, 170.0, 0.83333333333333304, 0.02, 0.51000000000000001, 0.029700000000000001, 0.00071000000000000002, 0.021000000000000001, 0.0076, 0.058700000000000002, 0.0184, 0.001, 0.001, 0.19, 0.255, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 774.96492999999998, 5.0, 0.0, 0.036959884032750301, 0.021659178617926, 0.02, 1.0, 0.00094367276997589099, 0.012388659243625, 0.000114596604507925}        // dynamic parameters
              ),
              amici::SecondOrderMode::none,                                  // o2mode
              std::vector<realtype>(13, 0.0),   // idlist
              std::vector<int>{},               // z2events
              true,                                        // pythonGenerated
              61,                       // ndxdotdp_explicit
              1,                       // ndxdotdx_explicit
              5                        // w_recursion_depth
          ) {
                 root_initial_values_ = std::vector<bool>(
                     rootInitialValues.begin(), rootInitialValues.end()
                 );
          }

    /**
     * @brief Clone this model instance.
     * @return A deep copy of this instance.
     */
    amici::Model *clone() const override {
        return new Model_icg_body(*this);
    }

    void fJrz(realtype *Jrz, const int iz, const realtype *p, const realtype *k, const realtype *rz, const realtype *sigmaz) override {}


    void fJy(realtype *Jy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my) override {
        Jy_icg_body(Jy, iy, p, k, y, sigmay, my);    
}


    void fJz(realtype *Jz, const int iz, const realtype *p, const realtype *k, const realtype *z, const realtype *sigmaz, const realtype *mz) override {}


    void fdJrzdsigma(realtype *dJrzdsigma, const int iz, const realtype *p, const realtype *k, const realtype *rz, const realtype *sigmaz) override {}


    void fdJrzdz(realtype *dJrzdz, const int iz, const realtype *p, const realtype *k, const realtype *rz, const realtype *sigmaz) override {}


    void fdJydsigma(realtype *dJydsigma, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my) override {
        dJydsigma_icg_body(dJydsigma, iy, p, k, y, sigmay, my);    
}


    void fdJzdsigma(realtype *dJzdsigma, const int iz, const realtype *p, const realtype *k, const realtype *z, const realtype *sigmaz, const realtype *mz) override {}


    void fdJzdz(realtype *dJzdz, const int iz, const realtype *p, const realtype *k, const realtype *z, const realtype *sigmaz, const double *mz) override {}


    /**
     * @brief model specific implementation of fdeltasx
     * @param deltaqB sensitivity update
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heaviside vector
     * @param ip sensitivity index
     * @param ie event index
     * @param xdot new model right hand side
     * @param xdot_old previous model right hand side
     * @param xB adjoint state
     */
    void fdeltaqB(realtype *deltaqB, const realtype t,
                  const realtype *x, const realtype *p,
                  const realtype *k, const realtype *h, const int ip,
                  const int ie, const realtype *xdot,
                  const realtype *xdot_old,
                  const realtype *xB) override {}

    void fdeltasx(realtype *deltasx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *sx, const realtype *stau, const realtype *tcl) override {}


    void fdeltax(double *deltax, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old) override {}


    /**
     * @brief model specific implementation of fdeltaxB
     * @param deltaxB adjoint state update
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heaviside vector
     * @param ie event index
     * @param xdot new model right hand side
     * @param xdot_old previous model right hand side
     * @param xB current adjoint state
     */
    void fdeltaxB(realtype *deltaxB, const realtype t,
                  const realtype *x, const realtype *p,
                  const realtype *k, const realtype *h, const int ie,
                  const realtype *xdot, const realtype *xdot_old,
                  const realtype *xB) override {}

    void fdrzdp(realtype *drzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {}


    void fdrzdx(realtype *drzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {}


    void fdsigmaydp(realtype *dsigmaydp, const realtype t, const realtype *p, const realtype *k, const realtype *y, const int ip) override {}


    void fdsigmaydy(realtype *dsigmaydy, const realtype t, const realtype *p, const realtype *k, const realtype *y) override {}


    void fdsigmazdp(realtype *dsigmazdp, const realtype t, const realtype *p, const realtype *k, const int ip) override {}


    void fdJydy(realtype *dJydy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my) override {
        dJydy_icg_body(dJydy, iy, p, k, y, sigmay, my);    
}

    void fdJydy_colptrs(SUNMatrixWrapper &colptrs, int index) override {        dJydy_colptrs_icg_body(colptrs, index);
    }

    void fdJydy_rowvals(SUNMatrixWrapper &rowvals, int index) override {        dJydy_rowvals_icg_body(rowvals, index);
    }


    void fdwdp(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *dtcldp) override {
        dwdp_icg_body(dwdp, t, x, p, k, h, w, tcl, dtcldp);    
}

    void fdwdp_colptrs(SUNMatrixWrapper &colptrs) override {        dwdp_colptrs_icg_body(colptrs);
    }

    void fdwdp_rowvals(SUNMatrixWrapper &rowvals) override {        dwdp_rowvals_icg_body(rowvals);
    }


    void fdwdx(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl) override {
        dwdx_icg_body(dwdx, t, x, p, k, h, w, tcl);    
}

    void fdwdx_colptrs(SUNMatrixWrapper &colptrs) override {        dwdx_colptrs_icg_body(colptrs);
    }

    void fdwdx_rowvals(SUNMatrixWrapper &rowvals) override {        dwdx_rowvals_icg_body(rowvals);
    }


    void fdwdw(realtype *dwdw, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl) override {
        dwdw_icg_body(dwdw, t, x, p, k, h, w, tcl);    
}

    void fdwdw_colptrs(SUNMatrixWrapper &colptrs) override {        dwdw_colptrs_icg_body(colptrs);
    }

    void fdwdw_rowvals(SUNMatrixWrapper &rowvals) override {        dwdw_rowvals_icg_body(rowvals);
    }


    void fdxdotdw(realtype *dxdotdw, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        dxdotdw_icg_body(dxdotdw, t, x, p, k, h, w);    
}

    void fdxdotdw_colptrs(SUNMatrixWrapper &colptrs) override {        dxdotdw_colptrs_icg_body(colptrs);
    }

    void fdxdotdw_rowvals(SUNMatrixWrapper &rowvals) override {        dxdotdw_rowvals_icg_body(rowvals);
    }


    void fdxdotdp_explicit(realtype *dxdotdp_explicit, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        dxdotdp_explicit_icg_body(dxdotdp_explicit, t, x, p, k, h, w);    
}

    void fdxdotdp_explicit_colptrs(SUNMatrixWrapper &colptrs) override {        dxdotdp_explicit_colptrs_icg_body(colptrs);
    }

    void fdxdotdp_explicit_rowvals(SUNMatrixWrapper &rowvals) override {        dxdotdp_explicit_rowvals_icg_body(rowvals);
    }


    void fdxdotdx_explicit(realtype *dxdotdx_explicit, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        dxdotdx_explicit_icg_body(dxdotdx_explicit, t, x, p, k, h, w);    
}

    void fdxdotdx_explicit_colptrs(SUNMatrixWrapper &colptrs) override {        dxdotdx_explicit_colptrs_icg_body(colptrs);
    }

    void fdxdotdx_explicit_rowvals(SUNMatrixWrapper &rowvals) override {        dxdotdx_explicit_rowvals_icg_body(rowvals);
    }


    void fdydx(realtype *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        dydx_icg_body(dydx, t, x, p, k, h, w, dwdx);    
}


    void fdydp(realtype *dydp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *tcl, const realtype *dtcldp) override {
        dydp_icg_body(dydp, t, x, p, k, h, ip, w, tcl, dtcldp);    
}


    void fdzdp(realtype *dzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {}


    void fdzdx(realtype *dzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {}


    void froot(realtype *root, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl) override {}


    void frz(realtype *rz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {}


    void fsigmay(realtype *sigmay, const realtype t, const realtype *p, const realtype *k, const realtype *y) override {
        sigmay_icg_body(sigmay, t, p, k, y);    
}


    void fsigmaz(realtype *sigmaz, const realtype t, const realtype *p, const realtype *k) override {}


    void fstau(realtype *stau, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *sx, const int ip, const int ie) override {}

    void fsx0(realtype *sx0, const realtype t, const realtype *x, const realtype *p, const realtype *k, const int ip) override {}

    void fsx0_fixedParameters(realtype *sx0_fixedParameters, const realtype t, const realtype *x0, const realtype *p, const realtype *k, const int ip, gsl::span<const int> reinitialization_state_idxs) override {}


    void fw(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl) override {
        w_icg_body(w, t, x, p, k, h, tcl);    
}


    void fx0(realtype *x0, const realtype t, const realtype *p, const realtype *k) override {
        x0_icg_body(x0, t, p, k);    
}


    void fx0_fixedParameters(realtype *x0_fixedParameters, const realtype t, const realtype *p, const realtype *k, gsl::span<const int> reinitialization_state_idxs) override {}


    void fxdot(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        xdot_icg_body(xdot, t, x, p, k, h, w);    
}


    void fy(realtype *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        y_icg_body(y, t, x, p, k, h, w);    
}


    void fz(realtype *z, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {}


    void fx_rdata(realtype *x_rdata, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k) override {
        x_rdata_icg_body(x_rdata, x, tcl, p, k);    
}


    void fx_solver(realtype *x_solver, const realtype *x_rdata) override {
        x_solver_icg_body(x_solver, x_rdata);    
}


    void ftotal_cl(realtype *total_cl, const realtype *x_rdata, const realtype *p, const realtype *k) override {
        total_cl_icg_body(total_cl, x_rdata, p, k);    
}


    void fdx_rdatadx_solver(realtype *dx_rdatadx_solver, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k) override {
        dx_rdatadx_solver_icg_body(dx_rdatadx_solver, x, tcl, p, k);    
}

    void fdx_rdatadx_solver_colptrs(SUNMatrixWrapper &colptrs) override {        dx_rdatadx_solver_colptrs_icg_body(colptrs);
    }

    void fdx_rdatadx_solver_rowvals(SUNMatrixWrapper &rowvals) override {        dx_rdatadx_solver_rowvals_icg_body(rowvals);
    }


    void fdx_rdatadp(realtype *dx_rdatadp, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k, const int ip) override {}


    void fdx_rdatadtcl(realtype *dx_rdatadtcl, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k) override {
        dx_rdatadtcl_icg_body(dx_rdatadtcl, x, tcl, p, k);    
}

    void fdx_rdatadtcl_colptrs(SUNMatrixWrapper &colptrs) override {        dx_rdatadtcl_colptrs_icg_body(colptrs);
    }

    void fdx_rdatadtcl_rowvals(SUNMatrixWrapper &rowvals) override {        dx_rdatadtcl_rowvals_icg_body(rowvals);
    }


    void fdtotal_cldp(realtype *dtotal_cldp, const realtype *x_rdata, const realtype *p, const realtype *k, const int ip) override {}


    void fdtotal_cldx_rdata(realtype *dtotal_cldx_rdata, const realtype *x_rdata, const realtype *p, const realtype *k, const realtype *tcl) override {
        dtotal_cldx_rdata_icg_body(dtotal_cldx_rdata, x_rdata, p, k, tcl);    
}

    void fdtotal_cldx_rdata_colptrs(SUNMatrixWrapper &colptrs) override {        dtotal_cldx_rdata_colptrs_icg_body(colptrs);
    }

    void fdtotal_cldx_rdata_rowvals(SUNMatrixWrapper &rowvals) override {        dtotal_cldx_rdata_rowvals_icg_body(rowvals);
    }


    std::string getName() const override {
        return "icg_body";
    }

    /**
     * @brief Get names of the model parameters
     * @return the names
     */
    std::vector<std::string> getParameterNames() const override {
        return std::vector<std::string>(parameterNames.begin(),
                                        parameterNames.end());
    }

    /**
     * @brief Get names of the model states
     * @return the names
     */
    std::vector<std::string> getStateNames() const override {
        return std::vector<std::string>(stateNames.begin(), stateNames.end());
    }

    /**
     * @brief Get names of the solver states
     * @return the names
     */
    std::vector<std::string> getStateNamesSolver() const override {
        std::vector<std::string> result;
        result.reserve(stateIdxsSolver.size());
        for(auto idx: stateIdxsSolver) {
            result.push_back(stateNames[idx]);
        }
        return result;
    }

    /**
     * @brief Get names of the fixed model parameters
     * @return the names
     */
    std::vector<std::string> getFixedParameterNames() const override {
        return std::vector<std::string>(fixedParameterNames.begin(),
                                        fixedParameterNames.end());
    }

    /**
     * @brief Get names of the observables
     * @return the names
     */
    std::vector<std::string> getObservableNames() const override {
        return std::vector<std::string>(observableNames.begin(),
                                        observableNames.end());
    }

    /**
     * @brief Get names of model expressions
     * @return Expression names
     */
    std::vector<std::string> getExpressionNames() const override {
        return std::vector<std::string>(expressionNames.begin(),
                                        expressionNames.end());
    }

    /**
     * @brief Get ids of the model parameters
     * @return the ids
     */
    std::vector<std::string> getParameterIds() const override {
        return std::vector<std::string>(parameterIds.begin(),
                                        parameterIds.end());
    }

    /**
     * @brief Get ids of the model states
     * @return the ids
     */
    std::vector<std::string> getStateIds() const override {
        return std::vector<std::string>(stateIds.begin(), stateIds.end());
    }

    /**
     * @brief Get ids of the solver states
     * @return the ids
     */
    std::vector<std::string> getStateIdsSolver() const override {
        std::vector<std::string> result;
        result.reserve(stateIdxsSolver.size());
        for(auto idx: stateIdxsSolver) {
            result.push_back(stateIds[idx]);
        }
        return result;
    }

    /**
     * @brief Get ids of the fixed model parameters
     * @return the ids
     */
    std::vector<std::string> getFixedParameterIds() const override {
        return std::vector<std::string>(fixedParameterIds.begin(),
                                        fixedParameterIds.end());
    }

    /**
     * @brief Get ids of the observables
     * @return the ids
     */
    std::vector<std::string> getObservableIds() const override {
        return std::vector<std::string>(observableIds.begin(),
                                        observableIds.end());
    }

    /**
     * @brief Get IDs of model expressions
     * @return Expression IDs
     */
    std::vector<std::string> getExpressionIds() const override {
        return std::vector<std::string>(expressionIds.begin(),
                                        expressionIds.end());
    }

    /**
     * @brief function indicating whether reinitialization of states depending
     * on fixed parameters is permissible
     * @return flag indicating whether reinitialization of states depending on
     * fixed parameters is permissible
     */
    bool isFixedParameterStateReinitializationAllowed() const override {
        return true;
    }

    /**
     * @brief returns the AMICI version that was used to generate the model
     * @return AMICI version string
     */
    std::string getAmiciVersion() const override {
        return "0.16.1";
    }

    /**
     * @brief returns the amici version that was used to generate the model
     * @return AMICI git commit hash
     */
    std::string getAmiciCommit() const override {
        return "unknown";
    }

    bool hasQuadraticLLH() const override {
        return true;
    }

    ObservableScaling getObservableScaling(int iy) const override {
        return observableScalings.at(iy);
    }
};


} // namespace model_icg_body

} // namespace amici

#endif /* _amici_TPL_MODELNAME_h */
