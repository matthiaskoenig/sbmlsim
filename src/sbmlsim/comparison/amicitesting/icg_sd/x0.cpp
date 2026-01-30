#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "p.h"

namespace amici {
namespace model_icg_sd {

void x0_icg_sd(realtype *x0, const realtype t, const realtype *p, const realtype *k){
    x0[9] = 0.01;
}

} // namespace model_icg_sd
} // namespace amici
