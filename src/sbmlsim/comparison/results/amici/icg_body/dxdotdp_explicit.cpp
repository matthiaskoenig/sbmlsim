#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"
#include "dxdotdp_explicit.h"

namespace amici {
namespace model_icg_body {

void dxdotdp_explicit_icg_body(realtype *dxdotdp_explicit, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    dxdot0_dBW = (-flux_Flow_ar_arre_icg + flux_Flow_re_ve_icg)/(std::pow(BW, 2)*Fblood*(1 - HCT)*(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1));  // dxdotdp_explicit[0]
    dxdot1_dBW = (-flux_Flow_ar_argi_icg + flux_Flow_gi_po_icg)/(std::pow(BW, 2)*FVgi*Fblood*(1 - HCT));  // dxdotdp_explicit[1]
    dxdot2_dBW = (-flux_Flow_arli_li_icg + flux_Flow_li_hv_icg - flux_Flow_po_li_icg + flux_LI__ICGIM)/(std::pow(BW, 2)*FVli*Fblood*(1 - HCT)*(1 - resection_rate));  // dxdotdp_explicit[2]
    dxdot3_dBW = (flux_Flow_lu_ar_icg - flux_Flow_ve_lu_icg)/(std::pow(BW, 2)*FVlu*Fblood*(1 - HCT));  // dxdotdp_explicit[3]
    dxdot4_dBW = -(-FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVve)*(flux_Flow_hv_ve_icg + flux_Flow_re_ve_icg - flux_Flow_ve_lu_icg + flux_iv_icg)/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2));  // dxdotdp_explicit[4]
    dxdot5_dBW = -(-FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVar)*(-flux_Flow_ar_argi_icg - flux_Flow_ar_arre_icg - flux_Flow_arli_hv_icg - flux_Flow_arli_li_icg + flux_Flow_lu_ar_icg)/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2));  // dxdotdp_explicit[5]
    dxdot6_dBW = -(-FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVpo)*(flux_Flow_gi_po_icg - flux_Flow_po_hv_icg - flux_Flow_po_li_icg)/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2));  // dxdotdp_explicit[6]
    dxdot7_dBW = -(-FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVhv)*(flux_Flow_arli_hv_icg - flux_Flow_hv_ve_icg + flux_Flow_li_hv_icg + flux_Flow_po_hv_icg)/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2));  // dxdotdp_explicit[7]
    dxdot9_dBW = (-flux_LI__ICGIM + flux_LI__ICGLI2CA)/(std::pow(BW, 2)*FVli*(1 - Fblood)*(1 - f_tissue_loss)*(1 - resection_rate));  // dxdotdp_explicit[8]
    dxdot10_dBW = (flux_LI__ICGLI2BI - flux_LI__ICGLI2CA)/(std::pow(BW, 2)*FVbi);  // dxdotdp_explicit[9]
    dxdot0_dFblood = (-flux_Flow_ar_arre_icg + flux_Flow_re_ve_icg)/(BW*std::pow(Fblood, 2)*(1 - HCT)*(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1));  // dxdotdp_explicit[10]
    dxdot1_dFblood = (-flux_Flow_ar_argi_icg + flux_Flow_gi_po_icg)/(BW*FVgi*std::pow(Fblood, 2)*(1 - HCT));  // dxdotdp_explicit[11]
    dxdot2_dFblood = (-flux_Flow_arli_li_icg + flux_Flow_li_hv_icg - flux_Flow_po_li_icg + flux_LI__ICGIM)/(BW*FVli*std::pow(Fblood, 2)*(1 - HCT)*(1 - resection_rate));  // dxdotdp_explicit[12]
    dxdot3_dFblood = (flux_Flow_lu_ar_icg - flux_Flow_ve_lu_icg)/(BW*FVlu*std::pow(Fblood, 2)*(1 - HCT));  // dxdotdp_explicit[13]
    dxdot4_dFblood = BW*FVve*(flux_Flow_hv_ve_icg + flux_Flow_re_ve_icg - flux_Flow_ve_lu_icg + flux_iv_icg)*(-FVar - FVhv - FVpo - FVve + 1)/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2)*(FVar + FVhv + FVpo + FVve));  // dxdotdp_explicit[14]
    dxdot5_dFblood = BW*FVar*(-FVar - FVhv - FVpo - FVve + 1)*(-flux_Flow_ar_argi_icg - flux_Flow_ar_arre_icg - flux_Flow_arli_hv_icg - flux_Flow_arli_li_icg + flux_Flow_lu_ar_icg)/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2)*(FVar + FVhv + FVpo + FVve));  // dxdotdp_explicit[15]
    dxdot6_dFblood = BW*FVpo*(flux_Flow_gi_po_icg - flux_Flow_po_hv_icg - flux_Flow_po_li_icg)*(-FVar - FVhv - FVpo - FVve + 1)/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2)*(FVar + FVhv + FVpo + FVve));  // dxdotdp_explicit[16]
    dxdot7_dFblood = BW*FVhv*(flux_Flow_arli_hv_icg - flux_Flow_hv_ve_icg + flux_Flow_li_hv_icg + flux_Flow_po_hv_icg)*(-FVar - FVhv - FVpo - FVve + 1)/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2)*(FVar + FVhv + FVpo + FVve));  // dxdotdp_explicit[17]
    dxdot9_dFblood = (flux_LI__ICGIM - flux_LI__ICGLI2CA)/(BW*FVli*std::pow(1 - Fblood, 2)*(1 - f_tissue_loss)*(1 - resection_rate));  // dxdotdp_explicit[18]
    dxdot0_dHCT = (flux_Flow_ar_arre_icg - flux_Flow_re_ve_icg)/(BW*Fblood*std::pow(1 - HCT, 2)*(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1));  // dxdotdp_explicit[19]
    dxdot1_dHCT = (flux_Flow_ar_argi_icg - flux_Flow_gi_po_icg)/(BW*FVgi*Fblood*std::pow(1 - HCT, 2));  // dxdotdp_explicit[20]
    dxdot2_dHCT = (flux_Flow_arli_li_icg - flux_Flow_li_hv_icg + flux_Flow_po_li_icg - flux_LI__ICGIM)/(BW*FVli*Fblood*std::pow(1 - HCT, 2)*(1 - resection_rate));  // dxdotdp_explicit[21]
    dxdot3_dHCT = (-flux_Flow_lu_ar_icg + flux_Flow_ve_lu_icg)/(BW*FVlu*Fblood*std::pow(1 - HCT, 2));  // dxdotdp_explicit[22]
    dxdot4_dHCT = (flux_Flow_hv_ve_icg + flux_Flow_re_ve_icg - flux_Flow_ve_lu_icg + flux_iv_icg)/(std::pow(1 - HCT, 2)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve));  // dxdotdp_explicit[23]
    dxdot5_dHCT = (-flux_Flow_ar_argi_icg - flux_Flow_ar_arre_icg - flux_Flow_arli_hv_icg - flux_Flow_arli_li_icg + flux_Flow_lu_ar_icg)/(std::pow(1 - HCT, 2)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));  // dxdotdp_explicit[24]
    dxdot6_dHCT = (flux_Flow_gi_po_icg - flux_Flow_po_hv_icg - flux_Flow_po_li_icg)/(std::pow(1 - HCT, 2)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo));  // dxdotdp_explicit[25]
    dxdot7_dHCT = (flux_Flow_arli_hv_icg - flux_Flow_hv_ve_icg + flux_Flow_li_hv_icg + flux_Flow_po_hv_icg)/(std::pow(1 - HCT, 2)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv));  // dxdotdp_explicit[26]
    dxdot0_dFVgi = (flux_Flow_ar_arre_icg - flux_Flow_re_ve_icg)/(BW*Fblood*(1 - HCT)*std::pow(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1, 2));  // dxdotdp_explicit[27]
    dxdot1_dFVgi = (-flux_Flow_ar_argi_icg + flux_Flow_gi_po_icg)/(BW*std::pow(FVgi, 2)*Fblood*(1 - HCT));  // dxdotdp_explicit[28]
    dxdot0_dFVbi = (flux_Flow_ar_arre_icg - flux_Flow_re_ve_icg)/(BW*Fblood*(1 - HCT)*std::pow(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1, 2));  // dxdotdp_explicit[29]
    dxdot10_dFVbi = (flux_LI__ICGLI2BI - flux_LI__ICGLI2CA)/(BW*std::pow(FVbi, 2));  // dxdotdp_explicit[30]
    dxdot0_dFVli = (flux_Flow_ar_arre_icg - flux_Flow_re_ve_icg)/(BW*Fblood*(1 - HCT)*std::pow(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1, 2));  // dxdotdp_explicit[31]
    dxdot2_dFVli = (-flux_Flow_arli_li_icg + flux_Flow_li_hv_icg - flux_Flow_po_li_icg + flux_LI__ICGIM)/(BW*std::pow(FVli, 2)*Fblood*(1 - HCT)*(1 - resection_rate));  // dxdotdp_explicit[32]
    dxdot9_dFVli = (-flux_LI__ICGIM + flux_LI__ICGLI2CA)/(BW*std::pow(FVli, 2)*(1 - Fblood)*(1 - f_tissue_loss)*(1 - resection_rate));  // dxdotdp_explicit[33]
    dxdot0_dFVlu = (flux_Flow_ar_arre_icg - flux_Flow_re_ve_icg)/(BW*Fblood*(1 - HCT)*std::pow(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1, 2));  // dxdotdp_explicit[34]
    dxdot3_dFVlu = (flux_Flow_lu_ar_icg - flux_Flow_ve_lu_icg)/(BW*std::pow(FVlu, 2)*Fblood*(1 - HCT));  // dxdotdp_explicit[35]
    dxdot0_dFVve = (flux_Flow_ar_arre_icg - flux_Flow_re_ve_icg)/(BW*Fblood*(1 - HCT)*std::pow(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1, 2));  // dxdotdp_explicit[36]
    dxdot4_dFVve = -(flux_Flow_hv_ve_icg + flux_Flow_re_ve_icg - flux_Flow_ve_lu_icg + flux_iv_icg)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2));  // dxdotdp_explicit[37]
    dxdot5_dFVve = -(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(-flux_Flow_ar_argi_icg - flux_Flow_ar_arre_icg - flux_Flow_arli_hv_icg - flux_Flow_arli_li_icg + flux_Flow_lu_ar_icg)/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2));  // dxdotdp_explicit[38]
    dxdot6_dFVve = -(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(flux_Flow_gi_po_icg - flux_Flow_po_hv_icg - flux_Flow_po_li_icg)/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2));  // dxdotdp_explicit[39]
    dxdot7_dFVve = -(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(flux_Flow_arli_hv_icg - flux_Flow_hv_ve_icg + flux_Flow_li_hv_icg + flux_Flow_po_hv_icg)/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2));  // dxdotdp_explicit[40]
    dxdot0_dFVar = (flux_Flow_ar_arre_icg - flux_Flow_re_ve_icg)/(BW*Fblood*(1 - HCT)*std::pow(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1, 2));  // dxdotdp_explicit[41]
    dxdot4_dFVar = -(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(flux_Flow_hv_ve_icg + flux_Flow_re_ve_icg - flux_Flow_ve_lu_icg + flux_iv_icg)/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2));  // dxdotdp_explicit[42]
    dxdot5_dFVar = -(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)*(-flux_Flow_ar_argi_icg - flux_Flow_ar_arre_icg - flux_Flow_arli_hv_icg - flux_Flow_arli_li_icg + flux_Flow_lu_ar_icg)/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2));  // dxdotdp_explicit[43]
    dxdot6_dFVar = -(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(flux_Flow_gi_po_icg - flux_Flow_po_hv_icg - flux_Flow_po_li_icg)/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2));  // dxdotdp_explicit[44]
    dxdot7_dFVar = -(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(flux_Flow_arli_hv_icg - flux_Flow_hv_ve_icg + flux_Flow_li_hv_icg + flux_Flow_po_hv_icg)/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2));  // dxdotdp_explicit[45]
    dxdot0_dFVpo = (flux_Flow_ar_arre_icg - flux_Flow_re_ve_icg)/(BW*Fblood*(1 - HCT)*std::pow(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1, 2));  // dxdotdp_explicit[46]
    dxdot4_dFVpo = -(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(flux_Flow_hv_ve_icg + flux_Flow_re_ve_icg - flux_Flow_ve_lu_icg + flux_iv_icg)/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2));  // dxdotdp_explicit[47]
    dxdot5_dFVpo = -(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(-flux_Flow_ar_argi_icg - flux_Flow_ar_arre_icg - flux_Flow_arli_hv_icg - flux_Flow_arli_li_icg + flux_Flow_lu_ar_icg)/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2));  // dxdotdp_explicit[48]
    dxdot6_dFVpo = -(flux_Flow_gi_po_icg - flux_Flow_po_hv_icg - flux_Flow_po_li_icg)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2));  // dxdotdp_explicit[49]
    dxdot7_dFVpo = -(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(flux_Flow_arli_hv_icg - flux_Flow_hv_ve_icg + flux_Flow_li_hv_icg + flux_Flow_po_hv_icg)/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2));  // dxdotdp_explicit[50]
    dxdot0_dFVhv = (flux_Flow_ar_arre_icg - flux_Flow_re_ve_icg)/(BW*Fblood*(1 - HCT)*std::pow(-FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1, 2));  // dxdotdp_explicit[51]
    dxdot4_dFVhv = -(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(flux_Flow_hv_ve_icg + flux_Flow_re_ve_icg - flux_Flow_ve_lu_icg + flux_iv_icg)/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2));  // dxdotdp_explicit[52]
    dxdot5_dFVhv = -(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(-flux_Flow_ar_argi_icg - flux_Flow_ar_arre_icg - flux_Flow_arli_hv_icg - flux_Flow_arli_li_icg + flux_Flow_lu_ar_icg)/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2));  // dxdotdp_explicit[53]
    dxdot6_dFVhv = -(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))*(flux_Flow_gi_po_icg - flux_Flow_po_hv_icg - flux_Flow_po_li_icg)/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2));  // dxdotdp_explicit[54]
    dxdot7_dFVhv = -(flux_Flow_arli_hv_icg - flux_Flow_hv_ve_icg + flux_Flow_li_hv_icg + flux_Flow_po_hv_icg)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2));  // dxdotdp_explicit[55]
    dxdot9_df_tissue_loss = (flux_LI__ICGIM - flux_LI__ICGLI2CA)/(BW*FVli*(1 - Fblood)*std::pow(1 - f_tissue_loss, 2)*(1 - resection_rate));  // dxdotdp_explicit[56]
    dxdot2_dresection_rate = (flux_Flow_arli_li_icg - flux_Flow_li_hv_icg + flux_Flow_po_li_icg - flux_LI__ICGIM)/(BW*FVli*Fblood*(1 - HCT)*std::pow(1 - resection_rate, 2));  // dxdotdp_explicit[57]
    dxdot9_dresection_rate = (flux_LI__ICGIM - flux_LI__ICGLI2CA)/(BW*FVli*(1 - Fblood)*(1 - f_tissue_loss)*std::pow(1 - resection_rate, 2));  // dxdotdp_explicit[58]
    dxdot11_dRi_icg = 1;  // dxdotdp_explicit[59]
    dxdot12_dRi_icg = 1;  // dxdotdp_explicit[60]
}

} // namespace model_icg_body
} // namespace amici
