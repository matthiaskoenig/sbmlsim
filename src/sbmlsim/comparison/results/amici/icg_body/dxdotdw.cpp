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
namespace model_icg_body {

void dxdotdw_icg_body(realtype *dxdotdw, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    dxdot11_dKi_icg = -IVDOSE_icg;  // dxdotdw[0]
    dxdot4_dflux_iv_icg = 1/((1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve));  // dxdotdw[1]
    dxdot0_dflux_Flow_ar_arre_icg = 1/(BW*Fblood*(1 - HCT)*(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1));  // dxdotdw[2]
    dxdot5_dflux_Flow_ar_arre_icg = -1/((1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));  // dxdotdw[3]
    dxdot0_dflux_Flow_re_ve_icg = -1/(BW*Fblood*(1 - HCT)*(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1));  // dxdotdw[4]
    dxdot4_dflux_Flow_re_ve_icg = 1/((1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve));  // dxdotdw[5]
    dxdot1_dflux_Flow_ar_argi_icg = 1/(BW*FVgi*Fblood*(1 - HCT));  // dxdotdw[6]
    dxdot5_dflux_Flow_ar_argi_icg = -1/((1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));  // dxdotdw[7]
    dxdot1_dflux_Flow_gi_po_icg = -1/(BW*FVgi*Fblood*(1 - HCT));  // dxdotdw[8]
    dxdot6_dflux_Flow_gi_po_icg = 1/((1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo));  // dxdotdw[9]
    dxdot2_dflux_Flow_arli_li_icg = 1/(BW*FVli*Fblood*(1 - HCT)*(1 - resection_rate));  // dxdotdw[10]
    dxdot5_dflux_Flow_arli_li_icg = -1/((1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));  // dxdotdw[11]
    dxdot5_dflux_Flow_arli_hv_icg = -1/((1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));  // dxdotdw[12]
    dxdot7_dflux_Flow_arli_hv_icg = 1/((1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv));  // dxdotdw[13]
    dxdot2_dflux_Flow_po_li_icg = 1/(BW*FVli*Fblood*(1 - HCT)*(1 - resection_rate));  // dxdotdw[14]
    dxdot6_dflux_Flow_po_li_icg = -1/((1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo));  // dxdotdw[15]
    dxdot6_dflux_Flow_po_hv_icg = -1/((1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo));  // dxdotdw[16]
    dxdot7_dflux_Flow_po_hv_icg = 1/((1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv));  // dxdotdw[17]
    dxdot2_dflux_Flow_li_hv_icg = -1/(BW*FVli*Fblood*(1 - HCT)*(1 - resection_rate));  // dxdotdw[18]
    dxdot7_dflux_Flow_li_hv_icg = 1/((1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv));  // dxdotdw[19]
    dxdot4_dflux_Flow_hv_ve_icg = 1/((1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve));  // dxdotdw[20]
    dxdot7_dflux_Flow_hv_ve_icg = -1/((1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv));  // dxdotdw[21]
    dxdot3_dflux_Flow_ve_lu_icg = 1/(BW*FVlu*Fblood*(1 - HCT));  // dxdotdw[22]
    dxdot4_dflux_Flow_ve_lu_icg = -1/((1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve));  // dxdotdw[23]
    dxdot3_dflux_Flow_lu_ar_icg = -1/(BW*FVlu*Fblood*(1 - HCT));  // dxdotdw[24]
    dxdot5_dflux_Flow_lu_ar_icg = 1/((1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));  // dxdotdw[25]
    dxdot2_dflux_LI__ICGIM = -1/(BW*FVli*Fblood*(1 - HCT)*(1 - resection_rate));  // dxdotdw[26]
    dxdot9_dflux_LI__ICGIM = 1/(BW*FVli*(1 - Fblood)*(1 - f_tissue_loss)*(1 - resection_rate));  // dxdotdw[27]
    dxdot9_dflux_LI__ICGLI2CA = -1/(BW*FVli*(1 - Fblood)*(1 - f_tissue_loss)*(1 - resection_rate));  // dxdotdw[28]
    dxdot10_dflux_LI__ICGLI2CA = 1/(BW*FVbi);  // dxdotdw[29]
    dxdot8_dflux_LI__ICGLI2BI = 1;  // dxdotdw[30]
    dxdot10_dflux_LI__ICGLI2BI = -1/(BW*FVbi);  // dxdotdw[31]
}

} // namespace model_icg_body
} // namespace amici
