#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_body {

static constexpr std::array<sunindextype, 1> dx_rdatadtcl_rowvals_icg_body_ = {
    9
};

void dx_rdatadtcl_rowvals_icg_body(SUNMatrixWrapper &dx_rdatadtcl){
    dx_rdatadtcl.set_indexvals(gsl::make_span(dx_rdatadtcl_rowvals_icg_body_));
}
} // namespace model_icg_body
} // namespace amici
