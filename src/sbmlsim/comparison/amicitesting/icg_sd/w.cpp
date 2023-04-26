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
namespace model_icg_sd {

void w_icg_sd(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl){
    LI__bil_ext = tcl_LI__bil_ext;  // w[0]
    BSA = 0.024264999999999998*std::pow(BW, 0.53779999999999994)*std::pow(HEIGHT, 0.39639999999999997);  // w[1]
    CLinfusion_icg = Ri_icg/(Mr_icg*(Cve_icg + 9.9999999999999998e-13));  // w[2]
    CO = BW*COBW*f_cardiac_output;  // w[3]
    ER_icg = (Car_icg - Chv_icg)/(Car_icg + 9.9999999999999995e-8);  // w[4]
    FVre = -FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1;  // w[5]
    Ki_icg = 41.579999999999998/ti_icg;  // w[6]
    Var = (1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);  // w[7]
    Vbi = BW*FVbi;  // w[8]
    Vgi = BW*FVgi;  // w[9]
    Vhv = (1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);  // w[10]
    Vli = BW*FVli*(1 - resection_rate);  // w[11]
    Vlu = BW*FVlu;  // w[12]
    Vpo = (1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);  // w[13]
    Vve = (1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);  // w[14]
    Aar_icg = Car_icg*(1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);  // w[15]
    Ahv_icg = Chv_icg*(1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);  // w[16]
    Apo_icg = Cpo_icg*(1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);  // w[17]
    Ave_icg = Cve_icg*(1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);  // w[18]
    QC = (3.0/50.0)*CO;  // w[19]
    Vgi_plasma = BW*FVgi*Fblood*(1 - HCT);  // w[20]
    Vgi_tissue = BW*FVgi*(1 - Fblood);  // w[21]
    Vli_plasma = BW*FVli*Fblood*(1 - HCT)*(1 - resection_rate);  // w[22]
    Vli_tissue = BW*FVli*(1 - Fblood)*(1 - f_tissue_loss)*(1 - resection_rate);  // w[23]
    Vlu_plasma = BW*FVlu*Fblood*(1 - HCT);  // w[24]
    Vlu_tissue = BW*FVlu*(1 - Fblood);  // w[25]
    Vre = BW*FVre;  // w[26]
    Agi_plasma_icg = Cgi_plasma_icg*Fblood*Vgi*(1 - HCT);  // w[27]
    Ali_plasma_icg = Cli_plasma_icg*Fblood*Vli*(1 - HCT);  // w[28]
    Alu_plasma_icg = Clu_plasma_icg*Fblood*Vlu*(1 - HCT);  // w[29]
    Mar_icg = Aar_icg*Mr_icg/((1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));  // w[30]
    Mhv_icg = Ahv_icg*Mr_icg/((1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv));  // w[31]
    Mpo_icg = Apo_icg*Mr_icg/((1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo));  // w[32]
    Mve_icg = Ave_icg*Mr_icg/((1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve));  // w[33]
    Qgi = FQgi*QC*f_bloodflow;  // w[34]
    Qh = FQh*QC*f_bloodflow*f_exercise;  // w[35]
    Qlu = FQlu*QC;  // w[36]
    Vre_plasma = BW*FVre*Fblood*(1 - HCT);  // w[37]
    Vre_tissue = BW*FVre*(1 - Fblood);  // w[38]
    Xar_icg = Aar_icg*Mr_icg;  // w[39]
    Xhv_icg = Ahv_icg*Mr_icg;  // w[40]
    Xpo_icg = Apo_icg*Mr_icg;  // w[41]
    Xve_icg = Ave_icg*Mr_icg;  // w[42]
    Are_plasma_icg = Cre_plasma_icg*Fblood*Vre*(1 - HCT);  // w[43]
    FQre = -Qh/Qlu + 1;  // w[44]
    Mgi_plasma_icg = Agi_plasma_icg*Mr_icg/(Fblood*Vgi*(1 - HCT));  // w[45]
    Mli_plasma_icg = Ali_plasma_icg*Mr_icg/(Fblood*Vli*(1 - HCT));  // w[46]
    Mlu_plasma_icg = Alu_plasma_icg*Mr_icg/(Fblood*Vlu*(1 - HCT));  // w[47]
    Qpo = Qgi;  // w[48]
    Xgi_plasma_icg = Agi_plasma_icg*Mr_icg;  // w[49]
    Xli_plasma_icg = Ali_plasma_icg*Mr_icg;  // w[50]
    Xlu_plasma_icg = Alu_plasma_icg*Mr_icg;  // w[51]
    Mre_plasma_icg = Are_plasma_icg*Mr_icg/(Fblood*Vre*(1 - HCT));  // w[52]
    Qha = Qh - Qpo;  // w[53]
    Qre = FQre*QC;  // w[54]
    Xre_plasma_icg = Are_plasma_icg*Mr_icg;  // w[55]
    flux_iv_icg = IVDOSE_icg*Ki_icg/Mr_icg;  // w[56]
    flux_Flow_ar_arre_icg = Car_icg*Qre;  // w[57]
    flux_Flow_re_ve_icg = Cre_plasma_icg*Qre;  // w[58]
    flux_Flow_ar_argi_icg = Car_icg*Qgi;  // w[59]
    flux_Flow_gi_po_icg = Cgi_plasma_icg*Qgi;  // w[60]
    flux_Flow_arli_li_icg = Car_icg*Qha*(1 - f_shunts);  // w[61]
    flux_Flow_arli_hv_icg = Car_icg*Qha*f_shunts;  // w[62]
    flux_Flow_po_li_icg = Cpo_icg*Qpo*(1 - f_shunts);  // w[63]
    flux_Flow_po_hv_icg = Cpo_icg*Qpo*f_shunts;  // w[64]
    flux_Flow_li_hv_icg = Cli_plasma_icg*(1 - f_shunts)*(Qha + Qpo);  // w[65]
    flux_Flow_hv_ve_icg = Chv_icg*Qh;  // w[66]
    flux_Flow_ve_lu_icg = Cve_icg*Qlu;  // w[67]
    flux_Flow_lu_ar_icg = Clu_plasma_icg*Qlu;  // w[68]
    flux_LI__ICGIM = Cli_plasma_icg*LI__ICGIM_Vmax*LI__f_oatp1b3*Vli*(1 - Fblood)*(1 - f_tissue_loss)/(Cli_plasma_icg + LI__ICGIM_Km*(1 + LI__bil_ext/LI__ICGIM_ki_bil));  // w[69]
    flux_LI__ICGLI2CA = LI__ICGLI2CA_Vmax*LI__icg*Vli*(1 - Fblood)*(1 - f_tissue_loss)/(LI__ICGLI2CA_Km + LI__icg);  // w[70]
    flux_LI__ICGLI2BI = LI__ICGLI2BI_Vmax*LI__icg_bi*Vli*(1 - Fblood)*(1 - f_tissue_loss);  // w[71]
}

} // namespace model_icg_sd
} // namespace amici
