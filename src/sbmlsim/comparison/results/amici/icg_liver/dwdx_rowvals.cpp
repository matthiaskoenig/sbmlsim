#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 3> dwdx_rowvals_icg_liver_ = {
    1, 2, 3
};

void dwdx_rowvals_icg_liver(SUNMatrixWrapper &dwdx){
    dwdx.set_indexvals(gsl::make_span(dwdx_rowvals_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
