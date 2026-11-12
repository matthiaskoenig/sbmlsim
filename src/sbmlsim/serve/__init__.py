"""The service which serves the interactive panels of the models.

A model declares its interactive surface in an annotation of its SBML
(`sbmlsim.panel`, `sbmlsim.panel_sbml`); this package serves that declaration
as the versioned payload `panel/1`, the app of a model, the pages of its
panels which its project declares (`sbmlsim.panel.App`, the payload `app/1`),
and runs the simulations behind them.

`runner` holds the two halves which need no HTTP: `ModelRegistry`, the models
a request may name, built once at startup from a mapping of model id to
project reference, and `PanelRunner`, which turns a panel and the values of
its knobs into the curves of a `PanelResult` with warm models. `data` reads
the study data a panel declares out of the data of its project, the measured
points under the curves, and needs no model either. `app` is the HTTP API over
them, `create_app(registry)`, `payload` the bodies it reads and answers and
`cache` the answers of the simulations by their hash, which an application
keeps so that the same request is simulated once. `jobs` is the other half of
a run: the simulation experiments of a model are minutes and not seconds, so a
`JobStore` takes them, runs one at a time in a thread of its own and lets a
caller follow the `Job` by its id and fetch the figures it collected.

Nothing of the package imports this one, so `import sbmlsim` imports neither
the service nor its dependencies; its modules are imported explicitly, e.g.
`from sbmlsim.serve.runner import ModelRegistry`. The service needs the extra
`web` (`pip install sbmlsim[web]`), which the package itself does not.
"""
