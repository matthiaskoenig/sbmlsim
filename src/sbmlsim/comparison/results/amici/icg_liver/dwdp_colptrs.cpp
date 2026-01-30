#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 8> dwdp_colptrs_icg_liver_ = {
    0, 1, 2, 3, 4, 5, 6, 7
};

void dwdp_colptrs_icg_liver(SUNMatrixWrapper &dwdp){
    dwdp.set_indexptrs(gsl::make_span(dwdp_colptrs_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
