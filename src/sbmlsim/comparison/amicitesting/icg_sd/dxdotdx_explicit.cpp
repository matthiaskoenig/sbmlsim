#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"
#include "dxdotdx_explicit.h"

namespace amici {
namespace model_icg_sd {

void dxdotdx_explicit_icg_sd(realtype *dxdotdx_explicit, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    dxdot11_dIVDOSE_icg = -Ki_icg;  // dxdotdx_explicit[0]
}

} // namespace model_icg_sd
} // namespace amici
