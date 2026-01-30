#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_body {

static constexpr std::array<sunindextype, 36> dwdp_colptrs_icg_body_ = {
    0, 0, 0, 27, 28, 29, 60, 84, 88, 90, 94, 98, 111, 124, 137, 150, 151, 152, 153, 153, 158, 162, 164, 165, 166, 169, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196
};

void dwdp_colptrs_icg_body(SUNMatrixWrapper &dwdp){
    dwdp.set_indexptrs(gsl::make_span(dwdp_colptrs_icg_body_));
}
} // namespace model_icg_body
} // namespace amici
