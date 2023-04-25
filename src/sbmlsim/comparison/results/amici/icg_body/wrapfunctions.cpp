#include "amici/model.h"
#include "wrapfunctions.h"
#include "icg_body.h"

namespace amici {
namespace generic_model {

std::unique_ptr<amici::Model> getModel() {
    return std::unique_ptr<amici::Model>(
        new amici::model_icg_body::Model_icg_body());
}


} // namespace generic_model

} // namespace amici
