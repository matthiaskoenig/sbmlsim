#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x_rdata.h"

namespace amici {
namespace model_icg_liver {

void x_solver_icg_liver(realtype *x_solver, const realtype *x_rdata){
    x_solver[0] = icg_ext;
    x_solver[1] = icg;
    x_solver[2] = icg_bi;
    x_solver[3] = icg_feces;
}

} // namespace model_icg_liver
} // namespace amici
