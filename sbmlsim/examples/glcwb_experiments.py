"""
Running typical simulation experiments with model.

Necessary to create the respective simulation experiments based on sbmlsim.
"""

import sbmlsim
from sbmlsim.tests.settings import MODEL_GLCWB

# TODO: whole-body simulation experiments (dose in mg, mg/kg, ...)
"""
[1] oral dose (single dose, at given start with given dose)
- oral glucose tolerance test (OGTT)

[2] bolus injection (at given time tstart, with given dose and given injection time, e.g. 5-10 min)
- c-peptide bolus
- ivGTT (glucose bolus)
- insulin injection

[3] constant infusion (for given period, tstart, tend)
- somatostatin infusion
- insulin infusion
- glucose infusion
- glucagon infusion

[4] clamping of substance (e.g. glucose, tstart, tend)
- glucose clamping (euglycemic clamp)
- insulin clamping (insulin clamp)

[5] combination experiments
- somatostatin infusion + c-peptide bolus
- hyperinsulinemic, euglycemic clamp
"""


# TODO: simulation experiments



def exlude_pkdb_parameter_filter(pid):
    """ Returns True if excluded, False otherwise

    :param pid:
    :return:
    """
    # TODO: implement
    # dose parameters
    if (pid.startswith("IVDOSE_")) or (pid.startswith("PODOSE_")):
        return True

    # physical parameters
    if (pid.startswith("Mr_")) or pid in ["R_PDB"]:
        return True
    return False


if __name__ == "__main__":

    # set

    path = model_path()
    r = sbmlsim.load_model(path)



