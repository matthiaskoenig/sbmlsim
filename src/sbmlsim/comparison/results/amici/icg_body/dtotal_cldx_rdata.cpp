#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x_rdata.h"
#include "p.h"
#include "tcl.h"
#include "dtotal_cldx_rdata.h"

namespace amici {
namespace model_icg_body {

void dtotal_cldx_rdata_icg_body(realtype *dtotal_cldx_rdata, const realtype *x_rdata, const realtype *p, const realtype *k, const realtype *tcl){
    dtotal_cl0_dLI__bil_ext = 1.0;  // dtotal_cldx_rdata[0]
}

} // namespace model_icg_body
} // namespace amici
