#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "p.h"
#include "y.h"
#include "sigmay.h"

namespace amici {
namespace model_icg_liver {

void sigmay_icg_liver(realtype *sigmay, const realtype t, const realtype *p, const realtype *k, const realtype *y){
    sigma_yicg_ext = 1.0;  // sigmay[0]
    sigma_ybil_ext = 1.0;  // sigmay[1]
    sigma_yicg = 1.0;  // sigmay[2]
    sigma_yicg_bi = 1.0;  // sigmay[3]
    sigma_yicg_feces = 1.0;  // sigmay[4]
    sigma_yVext = 1.0;  // sigmay[5]
    sigma_yVli = 1.0;  // sigmay[6]
    sigma_yVbi = 1.0;  // sigmay[7]
    sigma_yVfeces = 1.0;  // sigmay[8]
}

} // namespace model_icg_liver
} // namespace amici
