#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"
#include "tcl.h"
#include "dtcldp.h"
#include "dwdp.h"

namespace amici {
namespace model_icg_liver {

void dwdp_icg_liver(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *dtcldp){
    dflux_ICGIM_dICGIM_Vmax = 1.5*f_oatp1b3*icg_ext/(ICGIM_Km*(1 + bil_ext/ICGIM_ki_bil) + icg_ext);  // dwdp[0]
    dflux_ICGIM_dICGIM_Km = -1.5*ICGIM_Vmax*f_oatp1b3*icg_ext*(1 + bil_ext/ICGIM_ki_bil)/std::pow(ICGIM_Km*(1 + bil_ext/ICGIM_ki_bil) + icg_ext, 2);  // dwdp[1]
    dflux_ICGIM_dICGIM_ki_bil = 1.5*ICGIM_Km*ICGIM_Vmax*bil_ext*f_oatp1b3*icg_ext/(std::pow(ICGIM_ki_bil, 2)*std::pow(ICGIM_Km*(1 + bil_ext/ICGIM_ki_bil) + icg_ext, 2));  // dwdp[2]
    dflux_ICGIM_df_oatp1b3 = 1.5*ICGIM_Vmax*icg_ext/(ICGIM_Km*(1 + bil_ext/ICGIM_ki_bil) + icg_ext);  // dwdp[3]
    dflux_ICGLI2CA_dICGLI2CA_Vmax = 1.5*icg/(ICGLI2CA_Km + icg);  // dwdp[4]
    dflux_ICGLI2CA_dICGLI2CA_Km = -1.5*ICGLI2CA_Vmax*icg/std::pow(ICGLI2CA_Km + icg, 2);  // dwdp[5]
    dflux_ICGLI2BI_dICGLI2BI_Vmax = 1.5*icg_bi;  // dwdp[6]
}

} // namespace model_icg_liver
} // namespace amici
