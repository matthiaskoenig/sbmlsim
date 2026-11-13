# Interactive panels

A model declares its interactive surface in its own SBML: the knobs a reader turns, the curves which are drawn, the study data under them and the simulation they run. `python -m sbmlsim.serve` serves that declaration over HTTP and runs its simulations on warm models, so that a reader of a paper, a student in a lecture or a reviewer of a model moves a knob (an impairment class, a dose, a perfusion temperature) and sees the model answer.

The three parts are separate on purpose:

- **the declaration** travels with the model, as an annotation of its `<model>` element (`sbmlsim.panel`, `sbmlsim.panel_sbml`), so a panel is part of the SBML and of the COMBINE archive of a release and not of a checkout;
- **the service** is the python side (`sbmlsim.serve`, the extra `web`): a registry of models, the declarations as JSON and the simulations behind them;
- **the renderer** is javascript of a repository of its own, which draws a panel from the declaration alone.

They meet at three versioned JSON contracts, `panel/1`, `paneldata/1` and `app/1`, and at nothing else: see [The contracts](#the-contracts).

## The declaration in the SBML

The panels of a model are an `<annotation>` of its `<model>` element, in the namespace `https://sbmlsim.org/ns/panel/1`:

```xml
<model id="hctz" name="hydrochlorothiazide" timeUnits="min">
  <annotation>
    <panels xmlns="https://sbmlsim.org/ns/panel/1">
      <panel id="renal" title="Renal impairment" subtitle="hydrochlorothiazide after a single oral dose" xLabel="time [hr]">
        <simulation variant="hctz_body" end="1440" points="200" xUnit="hr">
          <preinit target="BW" value="75"/>
        </simulation>
        <knob id="dose" label="dose" target="PODOSE_hctz" unit="mg" kind="choice" default="25">
          <option label="12.5 mg" value="12.5"/>
          <option label="25 mg" value="25"/>
          <option label="50 mg" value="50"/>
        </knob>
        <knob id="egfr" label="renal function" target="f_renal_function" unit="dimensionless" kind="range" min="0.2" max="1.0" steps="9" default="1.0"/>
        <curve id="[Cve_hctz]" label="hydrochlorothiazide, plasma" unit="uM" scale="linear"/>
        <curve id="Aurine_hctz" label="hydrochlorothiazide, urine" unit="umol"/>
        <data id="beermann1976_urine" label="Beermann 1976 (5 mg, oral)" curve="Aurine_hctz" dataset="Beermann1976_Tab1A" x="time" y="value" conversion="1/(297.7 g/mole)">
          <select column="label" value="amount_urine_hctz5po_4"/>
        </data>
      </panel>
    </panels>
  </annotation>
</model>
```

A model declares as many `<panel>` elements as it has panels, each with one `<simulation>`, its knobs, its curves and the study data under them. Every attribute is the field of the declaration it carries, with the two exceptions `xLabel` and `xUnit`; an attribute which is not set means the default of its field, and a required text which is not set is a panel which cannot be read; the writer leaves out a text which is empty, which is read back as the default of its field, while a hand-written empty text is read as the empty text it is.

`<panel>`:

| attribute | what it is |
| --- | --- |
| `id` | the id a request names, required |
| `title` | the title of the panel, required |
| `subtitle` | the question the panel answers, empty without |
| `xLabel` | the label of the time axis, `time` without |

`<simulation>`, the one simulation every point of a panel runs:

| attribute | what it is |
| --- | --- |
| `variant` | the variant of the model (`Project.variants`), its default variant without |
| `end` | the end of the simulation, in the time unit of the model, required, greater than 0 and at most `10000` |
| `points` | the output steps, `200` without, from 2 to `2000`; a run answers the `points + 1` times from 0 to `end` |
| `xUnit` | the unit the time axis is delivered in, the time unit of the model without |

Its `<preinit target="BW" value="75"/>` children are the values of the pre-initialization which are no knob, e.g. a body weight; a knob writes its target over them.

`<knob>`, an entity of the model a reader sets:

| attribute | what it is |
| --- | --- |
| `id` | the key of its value in a request and the id of its axis in a grid, required |
| `label` | the knob as it is shown, required |
| `target` | the entity of the model it sets, a selection of the simulation, required |
| `unit` | the unit of its values, the unit of its target without |
| `kind` | `range` (the default) or `choice` |
| `min`, `max` | the bounds of a range knob, both required for one, `min` below `max` |
| `steps` | the points a grid over a range knob runs, `5` without, from 2 to `64` |
| `default` | the value the knob starts at, required; inside the bounds of a range knob and one of the choices of a choice knob |

A choice knob has its choices as `<option label="25 mg" value="25"/>` children, in their order.

`<curve>`, one output over the time:

| attribute | what it is |
| --- | --- |
| `id` | a selection of the model, e.g. `[Cve_hctz]`, and the key of its values in a result, required |
| `label` | its entry in the legend, required |
| `unit` | the unit of its values, the unit the model gives the selection without, see [Units](#units) |
| `scale` | `linear` (the default) or `log` |

`<data>`, the measured points drawn under a curve:

| attribute | what it is |
| --- | --- |
| `id` | the key of its values in a result, required; the ids of the series of a panel are distinct |
| `label` | the series as it is shown, e.g. the study it comes from, required |
| `curve` | the curve it belongs under, whose unit it is delivered in; its own `id` without |
| `dataset` | the id of a figure or a table of a study, e.g. `Beermann1976_Tab1A`, required; it is no path |
| `x`, `y` | the columns of the times and of the values, required |
| `yerr` | the column of the error of the values, none without |
| `conversion` | a quantity the values and their error are multiplied by, e.g. `1/(297.7 g/mole)` for the mass of a study which the model carries as a substance |
| `unit` | the unit the series is delivered in, the unit of its curve without, see [Units](#units) |

Its `<select column="label" value="amount_urine_hctz5po_4"/>` children are the rows of the dataset the series is, by the value of every column which selects them; a series without one is every row of its dataset.

The `dataset` is an id and never a path: a separator, a dot and a drive are refused where a panel is read, so a declaration cannot name a file outside of the data of the project of its model.

`sbmlsim.panel_sbml` is that annotation: `panels_from_sbml(sbml)` reads it and `write_panels(sbml_in, sbml_out, panels)` writes it into a model without changing anything else of it, each the inverse of the other for every field. A panel which cannot be read is reported as a `PanelProblem` of its id with its reason and never raised, so one malformed panel never takes a model down: `Project.panels_of(variant)` logs it and reads the others.

A model whose SBML we do not own is covered by `Project.panels`, a mapping of panel id to panel in the project of the model package: the panels of the annotation are read first and the ones of the project are laid over them by id. Both together are what the service serves.

### The limits

| limit | value | what it bounds |
| --- | --- | --- |
| `end` | 10000.0 | the end of the simulation of a panel, in the time unit of its model |
| `points` | 2000 | the output steps of that simulation, whose run answers one time more |
| `knobs` | 4 | the knobs of a panel, which are the axes one request may grid over |
| `grid` | 64 | the points of one axis, and the product of the axes of one request |
| `curves` | 16 | the curves of a panel |

They are the bounds of the declaration and of a request alike, so a request over them is refused where it is read, before a model is loaded, and the declaration of a panel carries them under `limits`: a renderer reads the limits of the endpoint it talks to instead of carrying its own.

## Units

A unit a panel declares is a promise its model has to keep, and this is the one rule behind it. A panel which does not follow it declares a unit its model cannot deliver, which is a failure of every request of that panel and not of the one declaration.

- **The unit of a curve is the unit it declares, else the unit the model gives that selection.** `<curve id="[Cve_hctz]" unit="uM"/>` is delivered in `uM`; `<curve id="Aurine_hctz"/>` is delivered in the unit the model carries `Aurine_hctz` in.
- **A declared unit which the values of the model cannot convert to is refused**, where the units of a panel are resolved and therefore alike on a run and on the study data: `<curve id="Cve_hctz" unit="uM"/>` on a model which carries `Cve_hctz` as an amount raises `The unit 'uM' of the curve 'Cve_hctz' is no unit of its values, which the model gives in 'mmol'.`, which the service logs with its traceback and answers with its fixed 500. The concentration of that species is the selection `[Cve_hctz]`, which is `mmol/litre` and converts.
- **The delivered unit is identical on `simulate` and on the study data**, unless a series declares a unit of its own. The `units` of a simulation and the `unit` of a series of `paneldata/1` of the same panel are the same text for one axis, and so are `units["x"]` and the `x_unit` of a series, so the points of a study and the curve above them share a scale.
- **"No unit" is `null` on both contracts.** A curve whose selection the model gives no unit, and a model whose time has no unit, deliver their values as they are; a dataset under them must carry no unit either.
- **A unit may be a unit id of the model**, e.g. `mmole_per_l`, because every unit of a panel is resolved in the unit registry of its model, the only registry which knows these ids.
- **The x** is the `xUnit` of the simulation, else the time unit of the model; `end` and `points` stay in the time unit of the model whatever `xUnit` is, so `end="1440"` of a model of minutes is one day delivered on an axis of hours.
- **A knob** takes its values in the unit it declares, else in the unit of its target; the values of a request and the `values` of an axis of a result are in that unit, which the result names.
- **A series of the study data** which declares a `unit` of its own is delivered in it as it is and is not checked against its curve, so a series may be drawn on a scale of its own; `conversion` is applied to the values and their error before. A series whose `id` is no curve of the panel is delivered in the unit of its own dataset.

## The app of a model

The pages above the panels are the declaration of the project of the model (`Project.app`, `sbmlsim.panel.App`) and no part of the SBML, because navigation is no model semantics: an app is a title, a subtitle, its pages, each a list of panel ids, the variants of the model a reader switches between and the citation, the DOI and the licenses of the provenance footer. A model app is therefore configuration and not code of its own.

What the declaration cannot check itself is checked when the service registers its models, before any request: a page which names a panel the model does not declare and a variant the project does not have are refused at startup, with a message which names them.

## The service

`sbmlsim.serve` is a FastAPI application over `Simulator.run`. It needs the extra `web` (fastapi, uvicorn and httpx), which `pip install "sbmlsim[web]"` installs and which the package itself does not need: nothing of sbmlsim imports the service, so `import sbmlsim` imports neither fastapi nor uvicorn.

### The command

```bash
python -m sbmlsim.serve --model ID=REFERENCE [--model ID=REFERENCE ...] [--host HOST] [--port PORT] [--static DIR] [--allow-origin ORIGIN ...] [--open]
```

| option | what it does |
| --- | --- |
| `--model ID=REFERENCE` | a model of the service: the id a request names and the project it is served from, `module:attribute` of a `sbmlsim.project.Project`; repeat it per model |
| `--host HOST` | the address the service binds, `127.0.0.1` without |
| `--port PORT` | the port, `8000` without |
| `--static DIR` | a directory served under the API, e.g. the slides or the pages which embed the panels |
| `--allow-origin ORIGIN` | an origin which may talk to the API from another one (CORS); repeat it per origin, none without |
| `--open` | open the browser on the service once it is up; off without |

The projects are loaded once, before the server starts, and the registry of the models is what a request reaches: a request names a model id of this command line and a panel id of that model and nothing else, never a module, a project reference or a path. A model is loaded and compiled when a panel of it is first simulated, or when the study data of a panel is first asked for, whose units its model resolves, and stays warm after that.

The address of the service is written on stdout, everything else is logged on stderr. The exit codes are the ones of [`python -m sbmlsim.run`](run.md): 0 when the server stopped, 2 for a bad command line, which argparse writes with its usage, and 4 for a model which is no project or whose app names something the model does not declare.

### The teaching mode: one command

A lecture, a talk or a page of the documentation is one command, which serves the pages and the API from one process:

```bash
pip install "sbmlsim[web]"
python -m sbmlsim.serve --model hctz=hctz_model.project:PROJECT --static slides --open
```

The deck is then `http://127.0.0.1:8000/` and the API is under the same origin, so a panel in the page talks to its own origin. That is not a convenience, it is what keeps the panels working in a browser of 2026:

- since Chrome 142 a request from a public `https` page to `http://localhost` is Local Network Access and asks the reader for a permission, which a lecture cannot rely on;
- Safari blocks the same request as mixed content, an `http` endpoint fetched from an `https` page, and never asks;
- a page which is served by the service itself is of the origin of the API, so neither applies and no CORS header is needed at all.

A page which is served from another origin, a deck on GitHub Pages or a course in Moodle, talks to a public `https` endpoint and needs `--allow-origin https://that.origin` for each of them. The service takes no credential: a browser sends no cookie and no authorization header of another origin to it.

### The routes

| route | what it answers |
| --- | --- |
| `GET /healthz` | that the service is up, with its version and the number of its models; it loads no model |
| `GET /api/models` | the models the service serves, in the order of the command line |
| `GET /api/models/{model}/app` | the app of the model, the contract `app/1`; a 404 for a model which declares none |
| `GET /api/models/{model}/panels` | the panels the model declares, their ids, titles and subtitles |
| `GET /api/models/{model}/panels/{panel}` | the declaration of one panel, the contract `panel/1` |
| `GET /api/models/{model}/panels/{panel}/data` | the study data behind its curves, the contract `paneldata/1` |
| `POST /api/models/{model}/panels/{panel}/simulate` | the curves of a run of the panel at the values of its knobs or over a grid of them |
| `GET /api/sim/{spec_hash}` | the stored answer of a simulation by its hash, as immutable |
| `POST /api/models/{model}/experiments` | a 202 with the job of a run of the simulation experiments of the model |
| `GET /api/jobs/{job_id}` | that job as it is now, with the figures of a finished one |
| `GET /api/jobs/{job_id}/figures/{path}` | one file of the directory of a job, e.g. a figure the job lists |
| `GET /docs`, `GET /openapi.json` | the OpenAPI document of the API, which FastAPI renders from the bodies |

`--static DIR` is mounted at `/` after these routes, so it shadows none of them.

The bodies below are the answers of a service of the hydrochlorothiazide model. Its panel is the one of the annotation above with a shorter simulation (120 min and two output steps), two doses, a coarser range knob and the default variant of the project, so that the arrays fit on this page.

The declaration of a panel is its `panel/1` payload:

```json
{
  "contract": "panel/1",
  "panel": {
    "id": "renal",
    "title": "Renal impairment",
    "subtitle": "hydrochlorothiazide after a single oral dose",
    "x_label": "time [hr]"
  },
  "axes": [
    {
      "id": "dose",
      "label": "dose",
      "target": "PODOSE_hctz",
      "unit": "mg",
      "kind": "choice",
      "min": null,
      "max": null,
      "steps": 5,
      "choices": [
        {"label": "12.5 mg", "value": 12.5},
        {"label": "25 mg", "value": 25.0}
      ],
      "default": 25.0
    },
    {
      "id": "egfr",
      "label": "renal function",
      "target": "f_renal_function",
      "unit": "dimensionless",
      "kind": "range",
      "min": 0.2,
      "max": 1.0,
      "steps": 3,
      "choices": [],
      "default": 1.0
    }
  ],
  "curves": [
    {"id": "[Cve_hctz]", "label": "hydrochlorothiazide, plasma", "unit": "uM", "scale": "linear"},
    {"id": "Aurine_hctz", "label": "hydrochlorothiazide, urine", "unit": "umol", "scale": "linear"}
  ],
  "data": [
    {
      "id": "beermann1976_urine",
      "label": "Beermann 1976 (5 mg, oral)",
      "curve": "Aurine_hctz",
      "dataset": "Beermann1976_Tab1A",
      "x": "time",
      "y": "value",
      "yerr": null,
      "sel": {"label": "amount_urine_hctz5po_4"},
      "conversion": "1/(297.7 g/mole)",
      "unit": null
    }
  ],
  "simulation": {
    "variant": null,
    "end": 120.0,
    "points": 2,
    "x_unit": "hr",
    "preinit": {"BW": 75.0}
  },
  "limits": {"end": 10000.0, "points": 2000, "knobs": 4, "grid": 64, "curves": 16}
}
```

A simulation takes the value of every knob a reader set and the points of every knob a grid runs over, and nothing else:

```json
{"values": {"dose": 25.0}, "grid": {"egfr": 3}}
```

A knob of the grid takes no value, a knob which a request leaves out runs at its `default`, and a body which names an id the panel does not declare, or a field the request does not have, is a 422 before a model is loaded. The answer is the grid, one row per point in C order:

```json
{
  "contract": "panel/1",
  "spec_hash": "31f8db7d3d6d14a46ad2a5d1642c36999a6ce89ea1ff9863bb2a97d553642e6d",
  "x": [0.0, 1.0, 2.0],
  "dims": [
    {"id": "egfr", "label": "renal function", "unit": "dimensionless", "values": [0.2, 0.6, 1.0]}
  ],
  "curves": {
    "[Cve_hctz]": [
      [1.727788792509682e-45, 0.4657270384445122, 0.7080096418038107],
      [1.7277887925096822e-45, 0.3916815929135674, 0.5462303310110749],
      [1.727788792509682e-45, 0.34200231622543875, 0.4516395144971838]
    ],
    "Aurine_hctz": [
      [1.6234091601443682e-69, 0.5849593608918757, 2.2902077861891277],
      [4.870219778211929e-69, 1.436210578900252, 5.26572987877097],
      [8.117020126691858e-69, 2.0211049712399904, 7.0914377715035295]
    ]
  },
  "units": {"x": "hr", "[Cve_hctz]": "uM", "Aurine_hctz": "umol"},
  "provenance": {
    "model": "hctz",
    "panel": "renal",
    "model_sha256": "e47e40b60ff28f8d3a12b0b99d09482ed3cba8e239dcb5cd9cc14cadae48f82c",
    "variant": null,
    "engine": "roadrunner",
    "sbmlsim": "0.22.0",
    "seconds": 0.07926412999950117,
    "created": "2026-11-12T22:41:35.090689+00:00"
  },
  "status": {"ok": true, "points": 3, "failed": []}
}
```

A grid is one request per figure and not one per movement of a knob: a renderer fetches it once and interpolates between its points while a knob moves. A point which did not integrate, a knob value the model does not simulate, is `null` in every curve and its flat index is in `status.failed`; the request stays a 200, since a model which does not integrate everywhere is a fact of the model and no error of the request.

The study data of the same panel is `paneldata/1`, on the axis of the curves and in their units:

```json
{
  "contract": "paneldata/1",
  "series": [
    {
      "id": "beermann1976_urine",
      "label": "Beermann 1976 (5 mg, oral)",
      "x": [0.0, 12.0, 96.0, 168.0],
      "y": [0.0, 10.110849848841115, 13.873026536781996, 13.889821968424588],
      "yerr": null,
      "x_unit": "hr",
      "unit": "umol",
      "count": [1, 1, 1, 1]
    }
  ]
}
```

A series is every row its `select` children choose, whatever the horizon of the simulation: the study above measured the urine for a week, the panel simulates two hours. A panel which declares no study data is answered with an empty `series` and loads no model; every other one pays the load a run of the panel pays as well, since the units of its series are the ones its model resolves. A row whose time or value is no number is no point of a series, so the payload carries no `NaN`, which JSON has not.

The app of the model is `app/1`:

```json
{
  "contract": "app/1",
  "title": "Hydrochlorothiazide",
  "subtitle": "a physiologically based model of hydrochlorothiazide",
  "pages": [{"id": "renal", "title": "Renal impairment", "panels": ["renal"]}],
  "variants": [],
  "citation": "Koenig et al. (2026)",
  "doi": "10.5281/zenodo.0000000",
  "licenses": ["CC BY-SA 4.0"]
}
```

### One simulation per request, once

Every answer of a simulation is stored under its `spec_hash`, the sha256 of the canonical JSON of the resolved request: the model id, the whole declaration of the panel, the variant, the sha256 of the SBML file which was simulated, the values of every knob and the points of every axis, the settings of the simulator of the project and the version of sbmlsim. A request which leaves a knob out therefore has the hash of the request which names its default, a presentation of many viewers who turn the same knob to the same value costs one simulation, and a changed panel or a changed model never answers under the hash of the one before it.

`GET /api/sim/{spec_hash}` serves a stored body by its hash alone, with `Cache-Control: public, max-age=31536000, immutable`, so a reader who follows a link of a hash fetches it once and a proxy may keep it. A hash which is not stored is a 404 of one fixed message whatever its shape, so the route tells a caller nothing but whether it holds the body it asks for. The cache keeps 256 responses, the one used least recently is evicted, and it belongs to the application: two services in one process never serve each other's bodies.

### The simulation experiments as jobs

A panel is a simulation of a second, which a request waits for. The simulation experiments of a model are minutes: they run their scans, evaluate their data and render their figures. They are therefore submitted and not awaited.

```json
{"experiments": ["Beermann1976", "Patel1984"]}
```

The names are groups of the project or experiment classes of it, at most sixteen of them, and `all` is every experiment of the model in one name; a name the model does not have is a 422 and submits nothing. The answer is a 202 with the job which was created:

```json
{
  "contract": "panel/1",
  "id": "6b2f1a0c9d4e4f7a8b1c2d3e4f5a6b7c",
  "model": "hctz",
  "experiments": ["Beermann1976", "Patel1984"],
  "status": "queued",
  "created": "2026-11-13T09:12:44.118302+00:00",
  "message": "",
  "figures": []
}
```

A caller follows it with `GET /api/jobs/{job_id}` until its `status` is `done` or `failed`, and a finished job lists the static figures its run collected, as the paths a request of one of them names:

```json
{"figures": ["_figures/Patel1984_Fig1.svg", "_figures/Patel1984_Fig_pk.svg"]}
```

`GET /api/jobs/{job_id}/figures/_figures/Patel1984_Fig1.svg` serves that file. A path is resolved against the directory of its job and is served only when it is a file inside it, so a path of `..`, an absolute path and a symbolic link which leaves the directory are the same 404 as a figure which does not exist, and no path of the host is reachable. The rule is the directory and not the list, so the `index.html` of the report of a run and its datasets are served under the same route.

The service runs one job at a time, in a thread of its own, and keeps at most eight of them in flight, the queued and the running one together: a submission above that is a 429. A run is serial, one process and no pool, so a job occupies one core of the service for its minutes; a job which failed says so in its `message`, which names no path and carries no traceback, while the traceback is logged. What a service keeps is bounded as well: the records of at most 32 finished jobs with their directories, the oldest of which is dropped when a job finishes above that, so a service which runs for weeks does not fill its disk with reports nobody reads any more.

The figures of a run are drawn without a display: `python -m sbmlsim.serve` sets `MPLBACKEND` to `Agg` unless the environment names a backend, as `python -m sbmlsim.run` does it. An interactive backend releases the objects of its toolkit in whatever thread the garbage collector happens to run in, and a job renders its figures in a thread of the server, which would end the service rather than one run.

A job loads models of its own, in its own thread: it runs the experiments of the model as a script would, and its runner is not the one which keeps the panels of the service warm. The store runs one job at a time, so two runs never load or simulate at once, but a load in a job is not serialized against a panel of the same model, which is answered in the thread pool of the server while the job runs. Taking the lock of the panel runner for the length of a run would block every panel of that model for minutes, which is the opposite of what the service is for.

The jobs belong to the application and not to a directory the service keeps: they live below a temporary directory which is created with the first job and removed when the service stops, so a service which never runs one writes nothing, a job of a service which was restarted is a 404, and nothing of a run outlives it.

### The errors

| status | when | body |
| --- | --- | --- |
| 404 | a model which is not served, a panel the model does not declare | `{"detail": "'nope' is no model of this service."}`, which names the id of the request and nothing of the service |
| 404 | a model which declares no app | `{"detail": "This model declares no app."}` |
| 404 | a hash which is not stored | `{"detail": "No response is stored under this hash."}` |
| 404 | a job the service does not hold, a path which is no figure of one | `{"detail": "No job is stored under this id."}`, `{"detail": "This job has no such figure."}` |
| 429 | a submission to a service which runs as many experiments as it keeps in flight | `{"detail": "The service runs as many experiments as it can at once."}` |
| 422 | a request over the limits, a body of an unknown field, a value which is no number | the `type`, the `loc` and the `msg` of at most ten errors, never the input which was sent |
| 422 | an id which is no knob of the panel | the ids and the knobs the panel has; it is checked before a model is loaded and before the cache is read |
| 422 | a name which is no experiment of the model | the names and the groups and experiment classes the model has; it is checked before the job is created |
| 500 | every failure of the service, e.g. a declaration whose unit does not convert | `{"detail": "The request failed in the service."}`, logged with its traceback |

No error body carries a traceback, a path or anything of the configuration of the service.

### What a public deployment must set

The defaults are the ones of the teaching mode. A service which is reachable from the network is a deliberate act and needs all of this:

- **`--host 0.0.0.0` behind a reverse proxy which terminates TLS.** The service speaks plain HTTP and binds the loopback address without `--host`; a page on `https` can only talk to an `https` endpoint.
- **`--allow-origin` for every origin which embeds the panels.** Without one the API answers its own origin alone, which is right for the local mode and wrong for a public one.
- **A rate limit and a body limit in the proxy.** An open simulation endpoint is remote compute. The limits above bound one request (and the models are the fixed list of `--model`), but nothing in the service bounds how many requests a caller makes; it has no authentication of its own either.
- **One process per service.** The warm models and the cache of the responses are of the process, so several workers would each load every model and keep a cache of their own. A model integrates under its own lock, so one model answers one simulation at a time while the other models and every other route keep answering; scale by running more services behind the proxy rather than more workers in one.
- **`GET /healthz` as the probe.** It answers without loading a model, so it stays cheap while a model compiles.
- **Memory.** Every model which is asked for stays loaded for the life of the process, one roadrunner instance per model and variant.

## The contracts

`panel/1` (the declaration and the answer of a simulation), `paneldata/1` (the study data) and `app/1` (the pages of a model) are the exchange point of this feature: the python side writes them and a renderer reads them, and neither knows anything else of the other.

What a renderer may rely on:

- **the version**, which every payload carries as `contract`: a renderer reads it and refuses a payload of a version it does not know, as `sbmlsim.panel.panel_from_payload` does on the python side;
- **plain JSON**, no type of sbmlsim and no python at all: objects, arrays, numbers, strings and `null`, where `null` is "no unit" and "no value" and nothing else;
- **the ids are data**: the ids of the models, panels, knobs, curves and series come from the payloads, so no model name and no entity of a model is built into a renderer;
- **the limits**, which the declaration of a panel carries under `limits`, so a renderer reads the bounds of the endpoint instead of carrying its own numbers;
- **one unit per axis**: the `units` of a simulation and the `unit` and `x_unit` of the series of the same panel state one unit unless a series declares a unit of its own, see [Units](#units);
- **the shape of a grid**: `curves[id]` is one row per point of `dims` in C order, `x` is the output times of every row, and a point which failed is `null` with its flat index in `status.failed`;
- **an added field keeps the version**: a payload may grow a field within `panel/1`, so a renderer ignores the fields it does not know, while the python readers of this package are strict and refuse a field they do not know; a field which is removed or changes its meaning is a new contract version, `panel/2`.

What a renderer must not rely on:

- **the python behind it**: no class, no module and no name of sbmlsim is part of a contract, and either side may be replaced on its own as long as it speaks it;
- **the hash**: `spec_hash` is an opaque string; what goes into it is the business of the service and changes with it;
- **the provenance values**: `seconds` and `created` are of the run which computed a body, and a stored body carries the ones of the run which computed it first;
- **the error texts**: the message of a 404 or a 422 is for a human, the status code is the contract;
- **the extra fields of a request**: a request body forbids what it does not declare, so a renderer sends `values` and `grid` and nothing else.

## Where the pieces are

| what | where |
| --- | --- |
| the declaration, the payloads and the limits | `sbmlsim.panel` |
| the annotation of the SBML, its reader and its writer | `sbmlsim.panel_sbml` |
| the panels of a model, the override of a project and its app | `Project.panels_of`, `Project.panels`, `Project.app`, see [Projects](project.md) |
| the registry of the models and the runner of their panels | `sbmlsim.serve.runner` |
| the HTTP API, its bodies and the cache of the answers | `sbmlsim.serve.app`, `sbmlsim.serve.payload`, `sbmlsim.serve.cache` |
| the study data behind the curves | `sbmlsim.serve.data` |
| the runs of the simulation experiments as jobs | `sbmlsim.serve.jobs` |
| the command line | `sbmlsim.serve.cli`, `python -m sbmlsim.serve` |
