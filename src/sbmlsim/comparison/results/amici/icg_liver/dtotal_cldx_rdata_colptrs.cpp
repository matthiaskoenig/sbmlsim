#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 6> dtotal_cldx_rdata_colptrs_icg_liver_ = {
    0, 0, 1, 1, 1, 1
};

void dtotal_cldx_rdata_colptrs_icg_liver(SUNMatrixWrapper &dtotal_cldx_rdata){
    dtotal_cldx_rdata.set_indexptrs(gsl::make_span(dtotal_cldx_rdata_colptrs_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
