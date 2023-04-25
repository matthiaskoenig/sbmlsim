#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_body {

static constexpr std::array<sunindextype, 14> dwdx_colptrs_icg_body_ = {
    0, 2, 4, 7, 9, 12, 18, 21, 24, 24, 25, 26, 27, 27
};

void dwdx_colptrs_icg_body(SUNMatrixWrapper &dwdx){
    dwdx.set_indexptrs(gsl::make_span(dwdx_colptrs_icg_body_));
}
} // namespace model_icg_body
} // namespace amici
