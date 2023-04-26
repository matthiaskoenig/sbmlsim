#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 5> dxdotdw_colptrs_icg_liver_ = {
    0, 0, 2, 4, 6
};

void dxdotdw_colptrs_icg_liver(SUNMatrixWrapper &dxdotdw){
    dxdotdw.set_indexptrs(gsl::make_span(dxdotdw_colptrs_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
