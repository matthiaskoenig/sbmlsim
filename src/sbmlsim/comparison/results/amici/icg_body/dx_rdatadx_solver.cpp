#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "tcl.h"
#include "p.h"
#include "dx_rdatadx_solver.h"

namespace amici {
namespace model_icg_body {

void dx_rdatadx_solver_icg_body(realtype *dx_rdatadx_solver, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k){
    dCre_plasma_icg_dx_solver0 = 1;  // dx_rdatadx_solver[0]
    dCgi_plasma_icg_dx_solver1 = 1;  // dx_rdatadx_solver[1]
    dCli_plasma_icg_dx_solver2 = 1;  // dx_rdatadx_solver[2]
    dClu_plasma_icg_dx_solver3 = 1;  // dx_rdatadx_solver[3]
    dCve_icg_dx_solver4 = 1;  // dx_rdatadx_solver[4]
    dCar_icg_dx_solver5 = 1;  // dx_rdatadx_solver[5]
    dCpo_icg_dx_solver6 = 1;  // dx_rdatadx_solver[6]
    dChv_icg_dx_solver7 = 1;  // dx_rdatadx_solver[7]
    dAfeces_icg_dx_solver8 = 1;  // dx_rdatadx_solver[8]
    dLI__icg_dx_solver9 = 1;  // dx_rdatadx_solver[9]
    dLI__icg_bi_dx_solver10 = 1;  // dx_rdatadx_solver[10]
    dIVDOSE_icg_dx_solver11 = 1;  // dx_rdatadx_solver[11]
    dcum_dose_icg_dx_solver12 = 1;  // dx_rdatadx_solver[12]
}

} // namespace model_icg_body
} // namespace amici
