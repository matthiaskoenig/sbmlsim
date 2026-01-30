#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "tcl.h"
#include "w.h"

namespace amici {
namespace model_icg_liver {

void w_icg_liver(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl){
    bil_ext = tcl_bil_ext;  // w[0]
    flux_ICGIM = 1.5*ICGIM_Vmax*f_oatp1b3*icg_ext/(ICGIM_Km*(1 + bil_ext/ICGIM_ki_bil) + icg_ext);  // w[1]
    flux_ICGLI2CA = 1.5*ICGLI2CA_Vmax*icg/(ICGLI2CA_Km + icg);  // w[2]
    flux_ICGLI2BI = 1.5*ICGLI2BI_Vmax*icg_bi;  // w[3]
}

} // namespace model_icg_liver
} // namespace amici
