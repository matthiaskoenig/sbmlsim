"""
Helpers for calculating model sensitivity.
"""

# FIXME: make this work with an Abstract model

def parameter_sensitivity_changeset(cls, r: roadrunner.RoadRunner,
                                    sensitivity: float = 0.1):
    """ Create changeset to calculate parameter sensitivity.

    :param r: model
    :param sensitivity: change for calculation of sensitivity
    :return: changeset
    """
    p_dict = cls._parameters_for_sensitivity(r)
    changeset = []
    for pid, value in p_dict.items():
        for change in [1.0 + sensitivity, 1.0 - sensitivity]:
            changeset.append(
                {pid: change * value}
            )
    return changeset


def _parameters_for_sensitivity(r, exclude_filter=None,
                                exclude_zero: bool = True,
                                zero_eps: float = 1E-8):
    """ Get parameter ids for sensitivity analysis.

    Values around current model state are used.

    :param r:
    :param exclude_filter: filter function to exclude parameters
    :param exclude_zero: exclude parameters which are zero
    :return:
    """
    doc = libsbml.readSBMLFromString(
        r.getSBML())  # type: libsbml.SBMLDocument
    model = doc.getModel()  # type: libsbml.Model

    # constant parameters
    pids_const = []
    for p in model.getListOfParameters():
        if p.getConstant() is True:
            pids_const.append(p.getId())

    # filter parameters
    parameters = OrderedDict()
    for pid in sorted(pids_const):
        if exclude_filter and exclude_filter(pid):
            continue

        value = r[pid]
        if exclude_zero:
            if np.abs(value) < zero_eps:
                continue

        parameters[pid] = value

    return parameters



def ensemble(sim: TimecourseSim, changeset: ChangeSet) -> List[TimecourseSim]:
    """ Creates an ensemble of timecourse by mixin the changeset in.

    :return: List[TimecourseSimulation]
    """
    sims = []
    for changes in changeset:
        # FIXME: not sure if this is doing the correct thing or custom implementation of copy and deepcopy needed
        sim_new = deepcopy(sim)
        # changes are mixed in the first timecourse
        tc = sim_new.timecourses[0]
        for key, value in changes.items():
            tc.add_change(key, value)
        sims.append(sim_new)

    return sims