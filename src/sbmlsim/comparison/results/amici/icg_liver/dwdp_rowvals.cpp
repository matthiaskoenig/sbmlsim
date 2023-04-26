#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 7> dwdp_rowvals_icg_liver_ = {
    1, 1, 1, 1, 2, 2, 3
};

void dwdp_rowvals_icg_liver(SUNMatrixWrapper &dwdp){
    dwdp.set_indexvals(gsl::make_span(dwdp_rowvals_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
