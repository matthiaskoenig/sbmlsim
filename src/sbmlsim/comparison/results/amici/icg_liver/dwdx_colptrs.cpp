#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 5> dwdx_colptrs_icg_liver_ = {
    0, 1, 2, 3, 3
};

void dwdx_colptrs_icg_liver(SUNMatrixWrapper &dwdx){
    dwdx.set_indexptrs(gsl::make_span(dwdx_colptrs_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
