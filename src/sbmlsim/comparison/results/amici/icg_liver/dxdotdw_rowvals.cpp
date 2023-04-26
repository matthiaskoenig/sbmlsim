#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 6> dxdotdw_rowvals_icg_liver_ = {
    0, 1, 1, 2, 2, 3
};

void dxdotdw_rowvals_icg_liver(SUNMatrixWrapper &dxdotdw){
    dxdotdw.set_indexvals(gsl::make_span(dxdotdw_rowvals_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
