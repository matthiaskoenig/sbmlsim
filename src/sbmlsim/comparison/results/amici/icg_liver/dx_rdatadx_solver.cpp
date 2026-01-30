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
namespace model_icg_liver {

void dx_rdatadx_solver_icg_liver(realtype *dx_rdatadx_solver, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k){
    dicg_ext_dx_solver0 = 1;  // dx_rdatadx_solver[0]
    dicg_dx_solver1 = 1;  // dx_rdatadx_solver[1]
    dicg_bi_dx_solver2 = 1;  // dx_rdatadx_solver[2]
    dicg_feces_dx_solver3 = 1;  // dx_rdatadx_solver[3]
}

} // namespace model_icg_liver
} // namespace amici
