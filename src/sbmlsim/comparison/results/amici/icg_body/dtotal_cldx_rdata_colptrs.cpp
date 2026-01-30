#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_body {

static constexpr std::array<sunindextype, 15> dtotal_cldx_rdata_colptrs_icg_body_ = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
};

void dtotal_cldx_rdata_colptrs_icg_body(SUNMatrixWrapper &dtotal_cldx_rdata){
    dtotal_cldx_rdata.set_indexptrs(gsl::make_span(dtotal_cldx_rdata_colptrs_icg_body_));
}
} // namespace model_icg_body
} // namespace amici
