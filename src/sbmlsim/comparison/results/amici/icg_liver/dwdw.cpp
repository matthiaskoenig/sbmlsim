#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"
#include "tcl.h"
#include "dwdw.h"

namespace amici {
namespace model_icg_liver {

void dwdw_icg_liver(realtype *dwdw, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl){
    dflux_ICGIM_dbil_ext = -1.5*ICGIM_Km*ICGIM_Vmax*f_oatp1b3*icg_ext/(ICGIM_ki_bil*std::pow(ICGIM_Km*(1 + bil_ext/ICGIM_ki_bil) + icg_ext, 2));  // dwdw[0]
}

} // namespace model_icg_liver
} // namespace amici
