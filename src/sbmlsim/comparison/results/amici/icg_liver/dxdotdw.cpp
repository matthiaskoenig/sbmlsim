#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"
#include "dxdotdw.h"

namespace amici {
namespace model_icg_liver {

void dxdotdw_icg_liver(realtype *dxdotdw, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    dxdot0_dflux_ICGIM = -0.25;  // dxdotdw[0]
    dxdot1_dflux_ICGIM = 0.66666666666666663;  // dxdotdw[1]
    dxdot1_dflux_ICGLI2CA = -0.66666666666666663;  // dxdotdw[2]
    dxdot2_dflux_ICGLI2CA = 1;  // dxdotdw[3]
    dxdot2_dflux_ICGLI2BI = -1;  // dxdotdw[4]
    dxdot3_dflux_ICGLI2BI = 1;  // dxdotdw[5]
}

} // namespace model_icg_liver
} // namespace amici
