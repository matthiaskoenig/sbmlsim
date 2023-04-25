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
namespace model_icg_body {

void xdot_icg_body(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    xdot0 = (flux_Flow_ar_arre_icg - flux_Flow_re_ve_icg)/(BW*Fblood*(1 - HCT)*(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1));  // xdot[0]
    xdot1 = (flux_Flow_ar_argi_icg - flux_Flow_gi_po_icg)/(BW*FVgi*Fblood*(1 - HCT));  // xdot[1]
    xdot2 = (flux_Flow_arli_li_icg - flux_Flow_li_hv_icg + flux_Flow_po_li_icg - flux_LI__ICGIM)/(BW*FVli*Fblood*(1 - HCT)*(1 - resection_rate));  // xdot[2]
    xdot3 = (-flux_Flow_lu_ar_icg + flux_Flow_ve_lu_icg)/(BW*FVlu*Fblood*(1 - HCT));  // xdot[3]
    xdot4 = (flux_Flow_hv_ve_icg + flux_Flow_re_ve_icg - flux_Flow_ve_lu_icg + flux_iv_icg)/((1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve));  // xdot[4]
    xdot5 = (-flux_Flow_ar_argi_icg - flux_Flow_ar_arre_icg - flux_Flow_arli_hv_icg - flux_Flow_arli_li_icg + flux_Flow_lu_ar_icg)/((1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));  // xdot[5]
    xdot6 = (flux_Flow_gi_po_icg - flux_Flow_po_hv_icg - flux_Flow_po_li_icg)/((1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo));  // xdot[6]
    xdot7 = (flux_Flow_arli_hv_icg - flux_Flow_hv_ve_icg + flux_Flow_li_hv_icg + flux_Flow_po_hv_icg)/((1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv));  // xdot[7]
    xdot8 = flux_LI__ICGLI2BI;  // xdot[8]
    xdot9 = (flux_LI__ICGIM - flux_LI__ICGLI2CA)/(BW*FVli*(1 - Fblood)*(1 - f_tissue_loss)*(1 - resection_rate));  // xdot[9]
    xdot10 = (-flux_LI__ICGLI2BI + flux_LI__ICGLI2CA)/(BW*FVbi);  // xdot[10]
    xdot11 = -IVDOSE_icg*Ki_icg + Ri_icg;  // xdot[11]
    xdot12 = Ri_icg;  // xdot[12]
}

} // namespace model_icg_body
} // namespace amici
