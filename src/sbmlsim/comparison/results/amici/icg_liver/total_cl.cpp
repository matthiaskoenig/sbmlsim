#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x_rdata.h"
#include "p.h"

namespace amici {
namespace model_icg_liver {

void total_cl_icg_liver(realtype *total_cl, const realtype *x_rdata, const realtype *p, const realtype *k){
    total_cl[0] = bil_ext;
}

} // namespace model_icg_liver
} // namespace amici
