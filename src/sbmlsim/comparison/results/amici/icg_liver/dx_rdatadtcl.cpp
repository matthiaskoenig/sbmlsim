#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "tcl.h"
#include "p.h"
#include "dx_rdatadtcl.h"

namespace amici {
namespace model_icg_liver {

void dx_rdatadtcl_icg_liver(realtype *dx_rdatadtcl, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k){
    dbil_ext_dtcl_bil_ext = 1;  // dx_rdatadtcl[0]
}

} // namespace model_icg_liver
} // namespace amici
