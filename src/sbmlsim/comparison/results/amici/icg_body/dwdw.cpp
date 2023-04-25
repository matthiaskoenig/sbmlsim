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
namespace model_icg_body {

void dwdw_icg_body(realtype *dwdw, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl){
    dflux_LI__ICGIM_dLI__bil_ext = -Cli_plasma_icg*LI__ICGIM_Km*LI__ICGIM_Vmax*LI__f_oatp1b3*Vli*(1 - Fblood)*(1 - f_tissue_loss)/(LI__ICGIM_ki_bil*std::pow(Cli_plasma_icg + LI__ICGIM_Km*(1 + LI__bil_ext/LI__ICGIM_ki_bil), 2));  // dwdw[0]
    dQC_dCO = 3.0/50.0;  // dwdw[1]
    dVre_dFVre = BW;  // dwdw[2]
    dVre_plasma_dFVre = BW*Fblood*(1 - HCT);  // dwdw[3]
    dVre_tissue_dFVre = BW*(1 - Fblood);  // dwdw[4]
    dflux_iv_icg_dKi_icg = IVDOSE_icg/Mr_icg;  // dwdw[5]
    dAgi_plasma_icg_dVgi = Cgi_plasma_icg*Fblood*(1 - HCT);  // dwdw[6]
    dMgi_plasma_icg_dVgi = -Agi_plasma_icg*Mr_icg/(Fblood*std::pow(Vgi, 2)*(1 - HCT));  // dwdw[7]
    dAli_plasma_icg_dVli = Cli_plasma_icg*Fblood*(1 - HCT);  // dwdw[8]
    dMli_plasma_icg_dVli = -Ali_plasma_icg*Mr_icg/(Fblood*std::pow(Vli, 2)*(1 - HCT));  // dwdw[9]
    dflux_LI__ICGIM_dVli = Cli_plasma_icg*LI__ICGIM_Vmax*LI__f_oatp1b3*(1 - Fblood)*(1 - f_tissue_loss)/(Cli_plasma_icg + LI__ICGIM_Km*(1 + LI__bil_ext/LI__ICGIM_ki_bil));  // dwdw[10]
    dflux_LI__ICGLI2CA_dVli = LI__ICGLI2CA_Vmax*LI__icg*(1 - Fblood)*(1 - f_tissue_loss)/(LI__ICGLI2CA_Km + LI__icg);  // dwdw[11]
    dflux_LI__ICGLI2BI_dVli = LI__ICGLI2BI_Vmax*LI__icg_bi*(1 - Fblood)*(1 - f_tissue_loss);  // dwdw[12]
    dAlu_plasma_icg_dVlu = Clu_plasma_icg*Fblood*(1 - HCT);  // dwdw[13]
    dMlu_plasma_icg_dVlu = -Alu_plasma_icg*Mr_icg/(Fblood*std::pow(Vlu, 2)*(1 - HCT));  // dwdw[14]
    dMar_icg_dAar_icg = Mr_icg/((1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));  // dwdw[15]
    dXar_icg_dAar_icg = Mr_icg;  // dwdw[16]
    dMhv_icg_dAhv_icg = Mr_icg/((1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv));  // dwdw[17]
    dXhv_icg_dAhv_icg = Mr_icg;  // dwdw[18]
    dMpo_icg_dApo_icg = Mr_icg/((1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo));  // dwdw[19]
    dXpo_icg_dApo_icg = Mr_icg;  // dwdw[20]
    dMve_icg_dAve_icg = Mr_icg/((1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve));  // dwdw[21]
    dXve_icg_dAve_icg = Mr_icg;  // dwdw[22]
    dQgi_dQC = FQgi*f_bloodflow;  // dwdw[23]
    dQh_dQC = FQh*f_bloodflow*f_exercise;  // dwdw[24]
    dQlu_dQC = FQlu;  // dwdw[25]
    dQre_dQC = FQre;  // dwdw[26]
    dAre_plasma_icg_dVre = Cre_plasma_icg*Fblood*(1 - HCT);  // dwdw[27]
    dMre_plasma_icg_dVre = -Are_plasma_icg*Mr_icg/(Fblood*std::pow(Vre, 2)*(1 - HCT));  // dwdw[28]
    dMgi_plasma_icg_dAgi_plasma_icg = Mr_icg/(Fblood*Vgi*(1 - HCT));  // dwdw[29]
    dXgi_plasma_icg_dAgi_plasma_icg = Mr_icg;  // dwdw[30]
    dMli_plasma_icg_dAli_plasma_icg = Mr_icg/(Fblood*Vli*(1 - HCT));  // dwdw[31]
    dXli_plasma_icg_dAli_plasma_icg = Mr_icg;  // dwdw[32]
    dMlu_plasma_icg_dAlu_plasma_icg = Mr_icg/(Fblood*Vlu*(1 - HCT));  // dwdw[33]
    dXlu_plasma_icg_dAlu_plasma_icg = Mr_icg;  // dwdw[34]
    dQpo_dQgi = 1;  // dwdw[35]
    dflux_Flow_ar_argi_icg_dQgi = Car_icg;  // dwdw[36]
    dflux_Flow_gi_po_icg_dQgi = Cgi_plasma_icg;  // dwdw[37]
    dFQre_dQh = -1/Qlu;  // dwdw[38]
    dQha_dQh = 1;  // dwdw[39]
    dflux_Flow_hv_ve_icg_dQh = Chv_icg;  // dwdw[40]
    dFQre_dQlu = Qh/std::pow(Qlu, 2);  // dwdw[41]
    dflux_Flow_ve_lu_icg_dQlu = Cve_icg;  // dwdw[42]
    dflux_Flow_lu_ar_icg_dQlu = Clu_plasma_icg;  // dwdw[43]
    dMre_plasma_icg_dAre_plasma_icg = Mr_icg/(Fblood*Vre*(1 - HCT));  // dwdw[44]
    dXre_plasma_icg_dAre_plasma_icg = Mr_icg;  // dwdw[45]
    dQre_dFQre = QC;  // dwdw[46]
    dQha_dQpo = -1;  // dwdw[47]
    dflux_Flow_po_li_icg_dQpo = Cpo_icg*(1 - f_shunts);  // dwdw[48]
    dflux_Flow_po_hv_icg_dQpo = Cpo_icg*f_shunts;  // dwdw[49]
    dflux_Flow_li_hv_icg_dQpo = Cli_plasma_icg*(1 - f_shunts);  // dwdw[50]
    dflux_Flow_arli_li_icg_dQha = Car_icg*(1 - f_shunts);  // dwdw[51]
    dflux_Flow_arli_hv_icg_dQha = Car_icg*f_shunts;  // dwdw[52]
    dflux_Flow_li_hv_icg_dQha = Cli_plasma_icg*(1 - f_shunts);  // dwdw[53]
    dflux_Flow_ar_arre_icg_dQre = Car_icg;  // dwdw[54]
    dflux_Flow_re_ve_icg_dQre = Cre_plasma_icg;  // dwdw[55]
}

} // namespace model_icg_body
} // namespace amici
