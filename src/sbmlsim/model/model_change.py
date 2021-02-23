"""Model changes.

Model changes are structural changes to the model structure.
Changes of values and initial conditions are encoded via
the changes instead.
"""
import logging

import roadrunner


logger = logging.getLogger(__name__)


class ModelChange(object):
    """ModelChange.

    Structural change to a model.
    """

    CLAMP_SPECIES = "clamp_species"

    @staticmethod
    def clamp_species(r: roadrunner.RoadRunner, species_id, formula=True, speed=1e4):
        """Clamp/free species to certain value or formula.

        This is only an approximative clamp, i.e. not working instantenious.
        Depending on the model kinetics different speed settings are required.
        FIXME: `time` cannot be used in formula due to https://github.com/sys-bio/roadrunner/issues/601
        FIXME: concentrations and amounts are not handled (uses native species setting
               i.e., amount or concentration definition.
        """
        if formula is None:
            # unclamping species.
            formula = False
        if formula is True:
            # clamping to current model value
            formula = r[species_id]

        selections = r.timeCourseSelections
        rid = f"{species_id}_clamp"

        if formula:
            r.addReaction(rid, [], [species_id], f"{speed}*({formula}-{species_id})")
            # update selections
            selections.append(rid)
            r.selections = selections
        else:
            r.removeReaction(rid)
            # update selections
            r.selections = [key for key in selections if key != rid]
