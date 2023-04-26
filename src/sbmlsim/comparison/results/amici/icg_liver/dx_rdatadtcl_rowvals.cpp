#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 1> dx_rdatadtcl_rowvals_icg_liver_ = {
    1
};

void dx_rdatadtcl_rowvals_icg_liver(SUNMatrixWrapper &dx_rdatadtcl){
    dx_rdatadtcl.set_indexvals(gsl::make_span(dx_rdatadtcl_rowvals_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
