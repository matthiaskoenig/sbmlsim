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
namespace model_icg_sd {

void dwdx_icg_sd(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl){
    dAre_plasma_icg_dCre_plasma_icg = Fblood*Vre*(1 - HCT);  // dwdx[0]
    dflux_Flow_re_ve_icg_dCre_plasma_icg = Qre;  // dwdx[1]
    dAgi_plasma_icg_dCgi_plasma_icg = Fblood*Vgi*(1 - HCT);  // dwdx[2]
    dflux_Flow_gi_po_icg_dCgi_plasma_icg = Qgi;  // dwdx[3]
    dAli_plasma_icg_dCli_plasma_icg = Fblood*Vli*(1 - HCT);  // dwdx[4]
    dflux_Flow_li_hv_icg_dCli_plasma_icg = (1 - f_shunts)*(Qha + Qpo);  // dwdx[5]
    dflux_LI__ICGIM_dCli_plasma_icg = -Cli_plasma_icg*LI__ICGIM_Vmax*LI__f_oatp1b3*Vli*(1 - Fblood)*(1 - f_tissue_loss)/std::pow(Cli_plasma_icg + LI__ICGIM_Km*(1 + LI__bil_ext/LI__ICGIM_ki_bil), 2) + LI__ICGIM_Vmax*LI__f_oatp1b3*Vli*(1 - Fblood)*(1 - f_tissue_loss)/(Cli_plasma_icg + LI__ICGIM_Km*(1 + LI__bil_ext/LI__ICGIM_ki_bil));  // dwdx[6]
    dAlu_plasma_icg_dClu_plasma_icg = Fblood*Vlu*(1 - HCT);  // dwdx[7]
    dflux_Flow_lu_ar_icg_dClu_plasma_icg = Qlu;  // dwdx[8]
    dCLinfusion_icg_dCve_icg = -Ri_icg/(Mr_icg*std::pow(Cve_icg + 9.9999999999999998e-13, 2));  // dwdx[9]
    dAve_icg_dCve_icg = (1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);  // dwdx[10]
    dflux_Flow_ve_lu_icg_dCve_icg = Qlu;  // dwdx[11]
    dER_icg_dCar_icg = (-Car_icg + Chv_icg)/std::pow(Car_icg + 9.9999999999999995e-8, 2) + 1.0/(Car_icg + 9.9999999999999995e-8);  // dwdx[12]
    dAar_icg_dCar_icg = (1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);  // dwdx[13]
    dflux_Flow_ar_arre_icg_dCar_icg = Qre;  // dwdx[14]
    dflux_Flow_ar_argi_icg_dCar_icg = Qgi;  // dwdx[15]
    dflux_Flow_arli_li_icg_dCar_icg = Qha*(1 - f_shunts);  // dwdx[16]
    dflux_Flow_arli_hv_icg_dCar_icg = Qha*f_shunts;  // dwdx[17]
    dApo_icg_dCpo_icg = (1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);  // dwdx[18]
    dflux_Flow_po_li_icg_dCpo_icg = Qpo*(1 - f_shunts);  // dwdx[19]
    dflux_Flow_po_hv_icg_dCpo_icg = Qpo*f_shunts;  // dwdx[20]
    dER_icg_dChv_icg = -1/(Car_icg + 9.9999999999999995e-8);  // dwdx[21]
    dAhv_icg_dChv_icg = (1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);  // dwdx[22]
    dflux_Flow_hv_ve_icg_dChv_icg = Qh;  // dwdx[23]
    dflux_LI__ICGLI2CA_dLI__icg = -LI__ICGLI2CA_Vmax*LI__icg*Vli*(1 - Fblood)*(1 - f_tissue_loss)/std::pow(LI__ICGLI2CA_Km + LI__icg, 2) + LI__ICGLI2CA_Vmax*Vli*(1 - Fblood)*(1 - f_tissue_loss)/(LI__ICGLI2CA_Km + LI__icg);  // dwdx[24]
    dflux_LI__ICGLI2BI_dLI__icg_bi = LI__ICGLI2BI_Vmax*Vli*(1 - Fblood)*(1 - f_tissue_loss);  // dwdx[25]
    dflux_iv_icg_dIVDOSE_icg = Ki_icg/Mr_icg;  // dwdx[26]
}

} // namespace model_icg_sd
} // namespace amici
