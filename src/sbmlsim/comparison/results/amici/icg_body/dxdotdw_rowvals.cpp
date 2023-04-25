#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_body {

static constexpr std::array<sunindextype, 32> dxdotdw_rowvals_icg_body_ = {
    11, 4, 0, 5, 0, 4, 1, 5, 1, 6, 2, 5, 5, 7, 2, 6, 6, 7, 2, 7, 4, 7, 3, 4, 3, 5, 2, 9, 9, 10, 8, 10
};

void dxdotdw_rowvals_icg_body(SUNMatrixWrapper &dxdotdw){
    dxdotdw.set_indexvals(gsl::make_span(dxdotdw_rowvals_icg_body_));
}
} // namespace model_icg_body
} // namespace amici
