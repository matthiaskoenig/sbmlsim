"""Projects of the tests of `verify`, referenced by `module:attribute`."""

from sbmlsim.project import CredibilitySettings
from tests.credibility_toy import PROJECT as CREDIBILITY_PROJECT
from tests.project_toy import PROJECT

#: the toy project with experiments of which one cannot be initialized
BROKEN = PROJECT.replace(experiments=".broken:BROKEN")

#: the credibility toy project with its evidence extension `arms`
WITH_ARMS = CREDIBILITY_PROJECT.replace(
    credibility_settings=CredibilitySettings(extensions=(".extensions:ARMS",))
)
