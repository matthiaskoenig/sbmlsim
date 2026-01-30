#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"
#include "tcl.h"
#include "dwdx.h"

namespace amici {
namespace model_icg_liver {

void dwdx_icg_liver(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl){
    dflux_ICGIM_dicg_ext = -1.5*ICGIM_Vmax*f_oatp1b3*icg_ext/std::pow(ICGIM_Km*(1 + bil_ext/ICGIM_ki_bil) + icg_ext, 2) + 1.5*ICGIM_Vmax*f_oatp1b3/(ICGIM_Km*(1 + bil_ext/ICGIM_ki_bil) + icg_ext);  // dwdx[0]
    dflux_ICGLI2CA_dicg = -1.5*ICGLI2CA_Vmax*icg/std::pow(ICGLI2CA_Km + icg, 2) + 1.5*ICGLI2CA_Vmax/(ICGLI2CA_Km + icg);  // dwdx[1]
    dflux_ICGLI2BI_dicg_bi = 1.5*ICGLI2BI_Vmax;  // dwdx[2]
}

} // namespace model_icg_liver
} // namespace amici
