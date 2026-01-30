#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 1> dtotal_cldx_rdata_rowvals_icg_liver_ = {
    0
};

void dtotal_cldx_rdata_rowvals_icg_liver(SUNMatrixWrapper &dtotal_cldx_rdata){
    dtotal_cldx_rdata.set_indexvals(gsl::make_span(dtotal_cldx_rdata_rowvals_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
