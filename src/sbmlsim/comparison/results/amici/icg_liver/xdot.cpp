#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"
#include "xdot.h"

namespace amici {
namespace model_icg_liver {

void xdot_icg_liver(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    xdot0 = -0.25*flux_ICGIM;  // xdot[0]
    xdot1 = 0.66666666666666663*flux_ICGIM - 0.66666666666666663*flux_ICGLI2CA;  // xdot[1]
    xdot2 = -flux_ICGLI2BI + flux_ICGLI2CA;  // xdot[2]
    xdot3 = flux_ICGLI2BI;  // xdot[3]
}

} // namespace model_icg_liver
} // namespace amici
