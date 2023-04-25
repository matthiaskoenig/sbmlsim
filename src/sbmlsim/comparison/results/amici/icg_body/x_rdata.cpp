#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "tcl.h"
#include "p.h"

namespace amici {
namespace model_icg_body {

void x_rdata_icg_body(realtype *x_rdata, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k){
    x_rdata[0] = Cre_plasma_icg;
    x_rdata[1] = Cgi_plasma_icg;
    x_rdata[2] = Cli_plasma_icg;
    x_rdata[3] = Clu_plasma_icg;
    x_rdata[4] = Cve_icg;
    x_rdata[5] = Car_icg;
    x_rdata[6] = Cpo_icg;
    x_rdata[7] = Chv_icg;
    x_rdata[8] = Afeces_icg;
    x_rdata[9] = tcl_LI__bil_ext;
    x_rdata[10] = LI__icg;
    x_rdata[11] = LI__icg_bi;
    x_rdata[12] = IVDOSE_icg;
    x_rdata[13] = cum_dose_icg;
}

} // namespace model_icg_body
} // namespace amici
