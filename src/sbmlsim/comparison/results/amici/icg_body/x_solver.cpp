#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x_rdata.h"

namespace amici {
namespace model_icg_body {

void x_solver_icg_body(realtype *x_solver, const realtype *x_rdata){
    x_solver[0] = Cre_plasma_icg;
    x_solver[1] = Cgi_plasma_icg;
    x_solver[2] = Cli_plasma_icg;
    x_solver[3] = Clu_plasma_icg;
    x_solver[4] = Cve_icg;
    x_solver[5] = Car_icg;
    x_solver[6] = Cpo_icg;
    x_solver[7] = Chv_icg;
    x_solver[8] = Afeces_icg;
    x_solver[9] = LI__icg;
    x_solver[10] = LI__icg_bi;
    x_solver[11] = IVDOSE_icg;
    x_solver[12] = cum_dose_icg;
}

} // namespace model_icg_body
} // namespace amici
