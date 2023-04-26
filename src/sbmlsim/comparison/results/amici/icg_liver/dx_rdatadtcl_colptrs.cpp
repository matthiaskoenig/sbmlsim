#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 2> dx_rdatadtcl_colptrs_icg_liver_ = {
    0, 1
};

void dx_rdatadtcl_colptrs_icg_liver(SUNMatrixWrapper &dx_rdatadtcl){
    dx_rdatadtcl.set_indexptrs(gsl::make_span(dx_rdatadtcl_colptrs_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
