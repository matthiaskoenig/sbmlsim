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
      <panel id="renal_plasma" title="Plasma concentration over the renal function" subtitle="hydrochlorothiazide after a single oral dose" xLabel="time [hr]" yLabel="plasma concentration">
        <simulation variant="hctz_body" end="1440" points="200" xUnit="hr">
          <preinit target="BW" value="75"/>
        </simulation>
        <knob id="dose" label="dose" target="PODOSE_hctz" unit="mg" kind="choice" default="25">
          <option label="5 mg" value="5"/>
          <option label="12.5 mg" value="12.5"/>
          <option label="25 mg" value="25"/>
          <option label="50 mg" value="50"/>
        </knob>
        <knob id="egfr" label="renal function" target="f_renal_function" unit="dimensionless" kind="choice" default="1.0">
          <option label="impaired" value="0.4"/>
          <option label="mildly impaired" value="0.7"/>
          <option label="normal" value="1.0"/>
        </knob>
        <curve id="[Cve_hctz]" label="plasma" unit="uM" scale="linear"/>
        <curve id="[Cki_plasma_hctz]" label="kidney" unit="uM"/>
      </panel>
      <panel id="renal_urine" title="Urinary excretion over the renal function" subtitle="hydrochlorothiazide after a single oral dose" xLabel="time [hr]" yLabel="amount excreted">
        <simulation variant="hctz_body" end="1440" points="200" xUnit="hr">
          <preinit target="BW" value="75"/>
        </simulation>
        <knob id="dose" label="dose" target="PODOSE_hctz" unit="mg" kind="choice" default="25">
          <option label="5 mg" value="5"/>
          <option label="12.5 mg" value="12.5"/>
          <option label="25 mg" value="25"/>
          <option label="50 mg" value="50"/>
        </knob>
        <knob id="egfr" label="renal function" target="f_renal_function" unit="dimensionless" kind="choice" default="1.0">
          <option label="impaired" value="0.4"/>
          <option label="mildly impaired" value="0.7"/>
          <option label="normal" value="1.0"/>
        </knob>
        <curve id="Aurine_hctz" label="urine" unit="umol"/>
        <data id="beermann1976_urine" label="Beermann 1976 (5 mg, oral)" curve="Aurine_hctz" dataset="Beermann1976_Tab1A" x="time" y="value" conversion="1/(297.7 g/mole)">
          <select column="label" value="amount_urine_hctz5po_4"/>
          <at knob="egfr" value="1.0"/>
          <at knob="dose" value="5"/>
        </data>
      </panel>
      <panel id="dose_plasma" title="Plasma concentration over the dose" xLabel="time [hr]" yLabel="plasma concentration">
        <simulation variant="hctz_body" end="1440" points="200" xUnit="hr">
          <preinit target="BW" value="75"/>
        </simulation>
        <knob id="dose" label="dose" target="PODOSE_hctz" unit="mg" kind="range" min="12.5" max="50" steps="4" default="25"/>
        <curve id="[Cve_hctz]" label="plasma" unit="uM"/>
      </panel>
      <panel id="patient_plasma" title="Plasma concentration of a patient" xLabel="time (hr)" yLabel="plasma concentration" yMin="0" yMax="2" xMax="24">
        <simulation end="1440" points="50" xUnit="hr">
          <preinit target="PODOSE_hctz" value="0"/>
          <preinit target="BW" value="75"/>
          <derived target="f_renal_function" formula="min(crcl / 110, 1)"/>
        </simulation>
        <knob id="crcl" label="creatinine clearance" unit="ml/min" kind="range" min="10" max="150" steps="8" default="110">
          <option label="normal" value="110"/>
          <option label="mild impairment" value="75"/>
          <option label="moderate impairment" value="45"/>
          <option label="severe impairment" value="20"/>
          <option label="kidney failure" value="10"/>
        </knob>
        <knob id="dose" label="dose" target="PODOSE_hctz" unit="mg" kind="choice" default="25">
          <option label="5 mg" value="5"/>
          <option label="12.5 mg" value="12.5"/>
          <option label="25 mg" value="25"/>
          <option label="50 mg" value="50"/>
        </knob>
        <curve id="[Cve_hctz]" label="plasma" unit="uM"/>
        <curve id="[Cki_plasma_hctz]" label="kidney" unit="uM"/>
      </panel>
    </panels>
  </annotation>
</model>
```

A model declares as many `<panel>` elements as it has panels, each with one `<simulation>`, its knobs, its curves and the study data under them. Every attribute is the field of the declaration it carries, with the camel case exceptions `xLabel`, `yLabel`, `yMin`, `yMax`, `xMax` and `xUnit` of the fields `x_label`, `y_label`, `y_min`, `y_max`, `x_max` and `x_unit`; an attribute which is not set means the default of its field, and a required text which is not set is a panel which cannot be read; the writer leaves out a text which is empty, which is read back as the default of its field, while a hand-written empty text is read as the empty text it is.

`<panel>`:

| attribute | what it is |
| --- | --- |
| `id` | the id a request names, required |
| `title` | the title of the panel, required |
| `subtitle` | the question the panel answers, empty without |
| `xLabel` | the label of the time axis, `time` without |
| `yLabel` | the label of the value axis, e.g. `plasma concentration`, empty without; a renderer names that axis after the curves then, and appends the unit they share in either case, so the unit is no part of it |
| `yMin`, `yMax` | the limits of the value axis, in the unit the curves of the panel are delivered in, which they then share; none without, which a renderer fits to the curves. `yMin` is below `yMax`, and both are positive on a panel whose curves declare the `log` scale |
| `xMax` | the end of the time axis, in the unit it is delivered in (`xUnit`, the time unit of the model without), greater than 0 and at most the end of the simulation; the end without |

The limits fix the axes, so two settings of the knobs are compared on one scale instead of an axis which follows each curve; the page `patient` of the example fixes them at the highest curve its knobs reach, rounded up, and its time axis at the day it simulates, `xMax="24"` hours of `end="1440"` minutes.

`<simulation>`, the one simulation every point of a panel runs:

| attribute | what it is |
| --- | --- |
| `variant` | the variant of the model (`Project.variants`), its default variant without |
| `end` | the end of the simulation, in the time unit of the model, required, greater than 0 and at most `10000` |
| `points` | the output steps, `200` without, from 2 to `2000`; a run answers the `points + 1` times from 0 to `end` |
| `xUnit` | the unit the time axis is delivered in, the time unit of the model without |

Its `<preinit target="BW" value="75"/>` children are the values of the pre-initialization which are no knob, e.g. a body weight; a knob writes its target over them. Its `<derived target="f_renal_function" formula="min(crcl / 110, 1)"/>` children are its derived targets, see [Derived targets](#derived-targets).

`<knob>`, an entity of the model a reader sets, or an input of the formulas of its derived targets:

| attribute | what it is |
| --- | --- |
| `id` | the key of its value in a request and the id of its axis in a grid, required |
| `label` | the knob as it is shown, required |
| `target` | the entity of the model it sets, a selection of the simulation; none for a knob which sets no entity itself and is an input of the formulas of the derived targets, which one of them reads |
| `unit` | the unit of its values, the unit of its target without |
| `kind` | `range` (the default) or `choice` |
| `min`, `max` | the bounds of a range knob, both required for one, `min` below `max` |
| `steps` | the points a grid over a range knob runs, `5` without, from 2 to `64` |
| `default` | the value the knob starts at, required; inside the bounds of a range knob and one of the choices of a choice knob |

A choice knob has its choices as `<option label="25 mg" value="25"/>` children, in their order. A grid over a choice knob runs its choices, so a choice knob has at most `MAX_GRID` (64) of them, which the declaration checks. A range knob may have them as well, as its named stops: labelled points of its range, each inside its bounds, which a renderer offers as a select of stops above the slider, e.g. the categories of a renal impairment on a slider of the creatinine clearance. The stops are no grid and no kind of their own: the grid of a range knob is its `steps` whatever its stops are, and its `default` need not be a stop.

`<curve>`, one output over the time:

| attribute | what it is |
| --- | --- |
| `id` | a selection of the model, e.g. `[Cve_hctz]`, and the key of its values in a result, required |
| `label` | its entry in the legend, required |
| `unit` | the unit of its values, the unit the model gives the selection without, see [Units](#units) |
| `scale` | `linear` (the default) or `log`, the scale the value axis opens on; the curves of a panel share that axis and declare one scale, a panel whose curves declare two is refused |

The curves of a panel share one axis, so a panel draws one quantity: a concentration of 0.45 and an amount of 47 on one axis is the concentration drawn flat on zero. A model which shows both declares two panels, which the example does with `renal_plasma`, the concentration in the venous plasma and in the plasma of the kidney in `uM`, and `renal_urine`, the amount excreted in `umol`. The label of a curve is then the site alone, since the title of the panel carries the substance.

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

Its `<at knob="dose" value="50"/>` children are the state of the knobs the series was measured under, by the value every one of them takes: a renderer draws the series only while the figure is at that state, so a panel never shows data of one condition under a curve of another, and a knob the children do not name is not constrained. The state which decides it is the one the drawn answer was run at and not the one the controls are in, since a control moves at once and the figure which answers it a round trip later: a series appears with the curve of its condition and never a moment before it. A knob of such a child is a knob of the panel of the kind `choice` and its value is one of its choices, or a range knob with named stops and its value is one of its stops, because a condition is one of a set of conditions and not a point of a continuum; a panel which conditions on a continuous input declares that knob as a choice knob or gives the range knob stops, and a range knob without stops is refused. The page `patient` of the example draws Beermann 1976 at the stop `normal` of its creatinine clearance, `<at knob="crcl" value="110"/>`. A series without such a child is drawn whatever the knobs are set to, which is right for data of a condition the panel does not vary. The example above shows both halves of that rule: the panel which draws Beermann 1976 declares the dose and the renal function as choices and names the state its series was measured at, the healthy volunteer who took 5 mg, while the panel of the dose alone, which draws no study data, varies the dose as the continuum of a range knob.

The `dataset` is an id and never a path: a separator, a dot and a drive are refused where a panel is read, so a declaration cannot name a file outside of the data of the project of its model.

`sbmlsim.panel_sbml` is that annotation: `panels_from_sbml(sbml)` reads it and `write_panels(sbml_in, sbml_out, panels)` writes it into a model without changing anything else of it, each the inverse of the other for every field. A panel which cannot be read is reported as a `PanelProblem` of its id with its reason and never raised, so one malformed panel never takes a model down: `Project.panels_of(variant)` logs it and reads the others.

A model whose SBML we do not own is covered by `Project.panels`, a mapping of panel id to panel in the project of the model package: the panels of the annotation are read first and the ones of the project are laid over them by id. Both together are what the service serves.

The declaration of a panel in a variant is the one of the SBML of that variant: each build of a model carries its own annotation, so a female build whose reference GFR differs from the male one declares its own stops of a renal impairment and its own study data, and nothing of a panel names a variant. `Project.panels` is the override of every variant and `Project.variant_panels`, a mapping of a variant to its panels, the override of one, laid over `Project.panels` for that variant, which is what a checkout whose SBML of a variant carries no annotation needs. `Project.panels_of(variant)` is the result: the annotated panels of the variant, then `panels`, then `variant_panels[variant]`, reading every SBML once. A model of a male and a female build of other reference creatinine clearances, each build its own SBML file, declares the page of a patient otherwise in its female build:

```py
FEMALE_PLASMA = PATIENT_PLASMA.model_copy(
    update={
        "title": "Plasma concentration of a patient (female)",
        "simulation": PATIENT_PLASMA.simulation.model_copy(
            update={"derived": {"f_renal_function": "min(crcl / 95, 1)"}}
        ),
    }
)

PROJECT = Project(
    name="drug",
    root=Path(__file__).parent,
    variants={
        "male": Path("models/drug_male.xml"),
        "female": Path("models/drug_female.xml"),
    },
    panels={"patient_plasma": PATIENT_PLASMA},
    variant_panels={"female": {"patient_plasma": FEMALE_PLASMA}},
    app=App(
        title="A drug",
        pages=(AppPage(id="patient", title="Patient", panels=("patient_plasma",)),),
        variants=(
            AppVariant(id="male", label="male"),
            AppVariant(id="female", label="female"),
        ),
    ),
)
```

The end to end tests of the renderer serve such a project of the HCTZ model, `examples.panels.hctz_variants:PROJECT_VARIANTS`, whose second variant is a copy of the one SBML file of the example, a device of the demo and of its tests, since a project gives every variant a file of its own.

### Derived targets

A knob sets one entity of the model, while an entity of a model is often a formula of the inputs a reader knows: the renal function of a model from a creatinine clearance, the activity of an enzyme from the activities of its two alleles. `<derived target="..." formula="..."/>` children of the `<simulation>` (`PanelSimulation.derived`) declare them: an entity of the model (an SBML `SId`) and a formula of the math of PEtab over the ids of the knobs of the panel and numbers, e.g. `min(crcl / 110, 1)` or `(a1 + a2) / 2 / 100`. A knob without a `target` is an input of such formulas alone.

- **The units are the formula's.** A formula sees the value of each knob as the number the knob carries, in the unit the knob declares and without a conversion, and its result is the value of the target in the unit of the model: `min(crcl / 110, 1)` reads a clearance in `ml/min` and gives the dimensionless renal function, so a formula states its conversion itself.
- **A derived target is set by its formula alone**: it is no target of a knob of the panel and no key of its `preinit`, which the declaration refuses.
- **The service compiles the formulas and a renderer never evaluates one.** When it starts, the service checks the formulas of every panel of every variant of a model, whether an app shows it or not: a formula which is no math, one which reads a symbol which is no knob of its panel and a knob without a target which no formula reads are refused with a message which names the model, the variant, the panel and the target or the knob, e.g. `The formula 'min(clearance / 110, 1)' of the derived target 'f_renal_function' of the panel 'patient_plasma' of the model 'hctz' in its variant 'default' reads ['clearance'], which are no knobs of the panel; ...`.
- **A target is an entity of the model.** When it starts, the service checks that the target of every knob, every derived target and every key of a `preinit` is a parameter, a species, a compartment or a species reference of an id of the model of each variant, or `[S]`, the concentration of a species `S`, read with libsbml from the SBML it reads for the panels anyway; a typo is refused with a message which names the model, the variant, the panel and the id, e.g. `The panel 'patient_plasma' of the model 'hctz' in its variant 'default' sets 'f_renal_fucntion', a derived target, which is no entity of the model: ...`, and not answered with a 500 on every request. A hierarchical model, whose entities are the ones of its flattened model, is not checked.
- **A run evaluates them with numpy.** A formula which reads a knob of the grid is a value of the dimension of the grid, evaluated at its points at once, and one which reads no knob of the grid is one value of the pre-initialization. The knobs of a grid are one dimension of the points of their product in C order, so a formula of two knobs of the grid has its value at every point, and the rows of the answer are in the C order of the knobs as they are without one.

### The limits

| limit | value | what it bounds |
| --- | --- | --- |
| `end` | 10000.0 | the end of the simulation of a panel, in the time unit of its model |
| `points` | 2000 | the output steps of that simulation, whose run answers one time more |
| `knobs` | 8 | the knobs of a panel, which are the axes one request may grid over; the work of a request is bounded by `grid` and `points`, and a knob which is not gridded is one value of the pre-initialization |
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
- **The limits of the value axis** (`yMin`, `yMax`) are in the unit the curves of the panel are delivered in, so the curves of a panel with such a limit are delivered in one unit, compared as a unit and not as a text (`uM` and `umol/l` are one); a panel whose curves are of different units and which declares one is refused where its units are resolved, alike on a run and on the study data, e.g. `The limits of the value axis of the panel 'renal_plasma' are in the unit its curves are delivered in, which they do not share: the curve '[Cve_hctz]' in 'uM', the curve '[Cki_plasma_hctz]' in 'nM'.`. **The end of the time axis** (`xMax`) is in the unit of the x and lies within the simulation, whose `end` is converted from the time unit of the model into it; one beyond it is refused there as well.
- **A series of the study data** which declares a `unit` of its own is delivered in it as it is and is not checked against its curve, so a series may be drawn on a scale of its own; `conversion` is applied to the values and their error before. A series whose `id` is no curve of the panel is delivered in the unit of its own dataset.

## The app of a model

The pages above the panels are the declaration of the project of the model (`Project.app`, `sbmlsim.panel.App`) and no part of the SBML, because navigation is no model semantics: an app is a title, a subtitle, its pages, each a list of panel ids with the scenarios a reader picks from, the variants of the model a reader switches between and the citation, the DOI and the licenses of the provenance footer. A model app is therefore configuration and not code of its own.

What the declaration cannot check itself is checked when the service registers its models, before any request: a page which names a panel the model does not declare and a variant the project does not have are refused at startup, with a message which names them. The declaration of a panel may differ by the variant, so the pages are checked in every variant the app is served in: the default variant of the project and every variant the app names. In each of them, every panel of every page is a panel of the variant, the shared knobs below are declared alike and the scenarios set values of their knobs; the app route answers these variants and refuses another one with a 422.

A knob id which more than one panel of a page declares is one knob of that page: the app shows one control for it above the panels and sets every panel which declares it from that control, so every plot of a page is of the state its controls show: a panel whose run failed says that it cannot draw that state rather than draw another one. The panels must therefore declare such an id alike, field for field, and two panels of one page which declare it differently are refused at startup, with a message which names the model, the page, the knob id, the two panels and the fields which differ, e.g. `The page 'renal' of the app of the model 'hctz' in its variant 'default' shows the panels 'renal_plasma' and 'renal_urine', which declare the knob 'dose' differently (default); the panels of a page share a knob they both declare, so they must declare it alike.`. The same id on different pages is two knobs, so the HCTZ app declares `dose` as a choice of tablets on its page `renal` and as a range on its page `dose`.

A page may declare scenarios (`AppPage.scenarios`, `sbmlsim.panel.Scenario`): named states of its knobs a reader picks from, an id distinct within the page, a label and the values of the knobs it sets, e.g. the example patients of a clinical question or the classes of an impairment. A reader who picks one sets every knob it names at once, and a knob it does not name keeps its value. When it registers its models the service checks that every knob a scenario names is a knob some panel of the page declares, and that its value is a value of that knob in every panel which declares it, inside the bounds of a range knob (a stop or not) and one of the choices of a choice knob, with a message which names the model, the variant, the page, the scenario, the knob and the panel. The shell owns every knob it sets, so a knob a scenario names is one control of the page and in `shared` even when one panel declares it, and the panel shows no control of its own for it. The page `patient` of the example has three patients, a healthy one at 25 mg, a moderate impairment at 25 mg and a severe one at 12.5 mg:

```py
AppPage(
    id="patient",
    title="Patient",
    panels=("patient_plasma", "patient_urine"),
    scenarios=(
        Scenario(
            id="healthy", label="healthy, 25 mg", values={"crcl": 110.0, "dose": 25.0}
        ),
        Scenario(
            id="moderate",
            label="moderate impairment, 25 mg",
            values={"crcl": 45.0, "dose": 25.0},
        ),
        Scenario(
            id="severe",
            label="severe impairment, 12.5 mg",
            values={"crcl": 20.0, "dose": 12.5},
        ),
    ),
)
```

Scenarios are of the app and not of the SBML, so they have no annotation.

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

The address of the service is written on stdout, everything else is logged on stderr. The exit codes are the ones of [`python -m sbmlsim.run`](run.md): 0 when the server stopped, 2 for a bad command line, which argparse writes with its usage, and 4 for a model which is no project, whose app names something a variant of the model does not declare, whose page declares one knob id differently on two panels or has a scenario of a value its knob does not take, or whose formula of a derived target is no math or reads no knob of its panel.

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
| `GET /api/models/{model}/app?variant=` | the app of the model, the contract `app/1`, with the shared knobs and the heads of the panels of a variant; a 404 for a model which declares none |
| `GET /api/models/{model}/panels` | the panels the default variant of the model declares, their ids, titles and subtitles |
| `GET /api/models/{model}/panels/{panel}?variant=` | the declaration of one panel in a variant, the contract `panel/1` |
| `GET /api/models/{model}/panels/{panel}/data?variant=` | the study data behind its curves in a variant, the contract `paneldata/1` |
| `POST /api/models/{model}/panels/{panel}/simulate` | the curves of a run of the panel at the values of its knobs or over a grid of them |
| `GET /api/sim/{spec_hash}` | the stored answer of a simulation by its hash, as immutable |
| `POST /api/models/{model}/experiments` | a 202 with the job of a run of the simulation experiments of the model |
| `GET /api/jobs/{job_id}` | that job as it is now, with the figures of a finished one |
| `GET /api/jobs/{job_id}/figures/{path}` | one file of the directory of a job, e.g. a figure the job lists |
| `GET /docs`, `GET /openapi.json` | the OpenAPI document of the API, which FastAPI renders from the bodies |

`--static DIR` is mounted at `/` after these routes, so it shadows none of them. The query `variant` is optional: without it a route answers the declaration of the default variant, and a variant the model does not have is a 422 which names the model and its variants.

The bodies below are the answers of a service of the hydrochlorothiazide model. The panels are the `renal_plasma` and the `renal_urine` of the annotation above with a shorter simulation (120 min and two output steps), two of their doses and the default variant of the project, so that the arrays fit on this page.

The declaration of a panel is its `panel/1` payload, here the one of `renal_urine`:

```json
{
  "contract": "panel/1",
  "panel": {
    "id": "renal_urine",
    "title": "Urinary excretion over the renal function",
    "subtitle": "hydrochlorothiazide after a single oral dose",
    "x_label": "time [hr]",
    "y_label": "amount excreted",
    "y_min": null,
    "y_max": null,
    "x_max": null
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
        {"label": "5 mg", "value": 5.0},
        {"label": "25 mg", "value": 25.0}
      ],
      "default": 25.0
    },
    {
      "id": "egfr",
      "label": "renal function",
      "target": "f_renal_function",
      "unit": "dimensionless",
      "kind": "choice",
      "min": null,
      "max": null,
      "steps": 5,
      "choices": [
        {"label": "impaired", "value": 0.4},
        {"label": "mildly impaired", "value": 0.7},
        {"label": "normal", "value": 1.0}
      ],
      "default": 1.0
    }
  ],
  "curves": [
    {"id": "Aurine_hctz", "label": "urine", "unit": "umol", "scale": "linear"}
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
      "at": {"egfr": 1.0, "dose": 5.0},
      "conversion": "1/(297.7 g/mole)",
      "unit": null
    }
  ],
  "simulation": {
    "variant": "default",
    "end": 120.0,
    "points": 2,
    "x_unit": "hr",
    "preinit": {"BW": 75.0},
    "derived": {}
  },
  "limits": {"end": 10000.0, "points": 2000, "knobs": 8, "grid": 64, "curves": 16}
}
```

A simulation takes the value of every knob a reader set, the points of every knob a grid runs over and the variant of the model, see below, and nothing else; the body below is a run of `renal_plasma`, whose two curves are one quantity in one unit:

```json
{"values": {"dose": 25.0}, "grid": {"egfr": 3}}
```

A knob of the grid takes no value, a knob which a request leaves out runs at its `default`, and a body which names an id the panel does not declare, or a field the request does not have, is a 422 before a model is loaded. So is a value its knob does not take in the declaration of the variant of the request - a value of a range knob outside `[min, max]`, a stop or not, a value of a choice knob which is none of its choices - and a grid over a choice knob of other points than its choices; the message names the knob, the value and what the knob takes, e.g. `The value 9.0 of the knob 'crcl' of the panel 'patient_plasma' is outside the bounds [10.0, 150.0].` The answer is the grid, one row per point in C order:

```json
{
  "contract": "panel/1",
  "spec_hash": "f5c84c9d5506be690bf9682ee5c07a61cfa38d6bb18dbed3d475d69d3a91b8e8",
  "x": [0.0, 1.0, 2.0],
  "dims": [
    {"id": "egfr", "label": "renal function", "unit": "dimensionless", "values": [0.4, 0.7, 1.0]}
  ],
  "curves": {
    "[Cve_hctz]": [
      [1.727788792509682e-45, 0.4247207865337207, 0.61515746136907],
      [1.727788792509682e-45, 0.3774826773795736, 0.5181364140723064],
      [1.727788792509682e-45, 0.34200231622543875, 0.4516395144971838]
    ],
    "[Cki_plasma_hctz]": [
      [6.348050654969098e-61, 0.3943226427434958, 0.5733082744820354],
      [6.348043125433088e-61, 0.33367381124285306, 0.4596838367900904],
      [6.348035595914937e-61, 0.28847389926841266, 0.3823005712958242]
    ]
  },
  "units": {"x": "hr", "[Cve_hctz]": "uM", "[Cki_plasma_hctz]": "uM"},
  "provenance": {
    "model": "hctz",
    "panel": "renal_plasma",
    "model_sha256": "e47e40b60ff28f8d3a12b0b99d09482ed3cba8e239dcb5cd9cc14cadae48f82c",
    "variant": "default",
    "engine": "roadrunner",
    "sbmlsim": "0.22.0",
    "seconds": 0.2634315009927377,
    "created": "2026-11-13T06:49:03.777256+00:00"
  },
  "status": {"ok": true, "points": 3, "failed": []}
}
```

A grid is one request per figure and not one per movement of a knob: a renderer fetches it once and interpolates between its points while a knob moves. A point which did not integrate, a knob value the model does not simulate, is `null` in every curve and its flat index is in `status.failed`; the request stays a 200, since a model which does not integrate everywhere is a fact of the model and no error of the request.

A request may also name the `variant` of the model to simulate (`Project.variants`), which an app switches between, e.g. `{"values": {"crcl": 45.0}, "grid": {}, "variant": "female"}`: the declaration of the panel in that variant runs on the SBML of that variant, so the knobs, the derived targets and the curves are the ones that variant declares. Without one the declaration of the default variant runs on the variant it declares, else on the default variant of the project. A variant the project does not have is a 422 which names the model and its variants, and a panel the variant does not declare is a 404. The `spec_hash` covers the variant and the whole declaration, and `provenance.variant` says which one ran. The declaration and the data routes take the same variant as the query `variant`, so a renderer which switches the variant reads the declaration, the data and the curves of one variant.

The study data of `renal_urine` is `paneldata/1`, on the axis of its curves and in their units:

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
  "pages": [{"id": "renal", "title": "Renal impairment", "panels": ["renal_plasma", "renal_urine"], "scenarios": []}],
  "variants": [],
  "citation": "Koenig et al. (2026)",
  "doi": "10.5281/zenodo.0000000",
  "licenses": ["CC BY-SA 4.0"],
  "shared": {
    "renal": [
      {
        "knob": {"id": "egfr", "label": "renal function", "target": "f_renal_function", "unit": "dimensionless", "kind": "choice", "min": null, "max": null, "steps": 5, "choices": [{"label": "impaired", "value": 0.4}, {"label": "mildly impaired", "value": 0.7}, {"label": "normal", "value": 1.0}], "default": 1.0},
        "panels": [
          {"id": "renal_plasma", "title": "Plasma concentration over the renal function"},
          {"id": "renal_urine", "title": "Urinary excretion over the renal function"}
        ]
      }
    ]
  },
  "heads": {
    "renal": [
      {"id": "renal_plasma", "title": "Plasma concentration over the renal function", "subtitle": "..."},
      {"id": "renal_urine", "title": "Urinary excretion over the renal function", "subtitle": "..."}
    ]
  },
  "variant": "default"
}
```

Next to the declaration of the app the body carries `shared`, which the service derives from the panels and which is no part of the declaration (neither of `Project.app` nor of an annotation): for every page, the knobs the shell of the page owns, the ones more than one of its panels declares and the ones a scenario of the page sets, each as the item of the `axes` of the panels which declare it and with the ids and the titles of those panels, in the order the first panel which declares one lists them. Every page has an entry, an empty list for a page whose panels share no knob and which has no scenario. With `?variant=` they are worked out of the panels of that variant, as `heads` are. The service works them out once, when it registers its models, which is also where it refuses a page whose panels declare one id differently, so a shell reads which controls a page shares from the app and never from the declarations of its panels. The example is shortened to one shared knob; the HCTZ page `renal` shares the dose as well.

Next to it the body carries `heads`, derived from the panels as `shared` is and no part of the declaration either: for every page, its panels in the order the page shows them, each with its id, its title and its subtitle as its declaration states them. A shell lays out the head of each card with them before the panel has read its own declaration, so the head row of a row of cards is as tall as its tallest head from the first frame and a card does not move when the head of its panel arrives; the subtitles are elided in the example.

Last, the body names the `variant` it answers: the one a request names with `?variant=`, else the default variant of the project, which a request without one is answered with. A shell which reads the app knows by it which variant its controls and heads are of, so it needs no assumption about which variant the app lists first, and it refuses an answer of another variant than it asked for.

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
| 422 | a value a knob does not take, or a grid over a choice knob of other points than its choices | the knob, the value and the bounds or the choices of the knob in the declaration of the variant of the request; it is checked before a model is loaded and before the cache is read |
| 422 | a variant the model does not have, or for the app one the app does not name | the model and its variants; it is checked before a model is loaded |
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
- **the extra fields of a request**: a request body forbids what it does not declare, so a renderer sends `values`, `grid` and `variant` and nothing else;
- **a formula**: the derived targets of a panel are evaluated by the service, so a renderer reads them as declared text and never evaluates one.

## Where the pieces are

| what | where |
| --- | --- |
| the declaration, the payloads and the limits | `sbmlsim.panel` |
| the annotation of the SBML, its reader and its writer | `sbmlsim.panel_sbml` |
| the panels of a model, the overrides of a project and its app | `Project.panels_of`, `Project.panels`, `Project.variant_panels`, `Project.app`, see [Projects](project.md) |
| the demo of every feature, which the end to end tests of the renderer serve | `examples.panels.hctz:PROJECT`; for the variants `examples.panels.hctz_variants:PROJECT_VARIANTS`, a device of the demo and no pattern to follow |
| the registry of the models and the runner of their panels | `sbmlsim.serve.runner` |
| the HTTP API, its bodies and the cache of the answers | `sbmlsim.serve.app`, `sbmlsim.serve.payload`, `sbmlsim.serve.cache` |
| the study data behind the curves | `sbmlsim.serve.data` |
| the runs of the simulation experiments as jobs | `sbmlsim.serve.jobs` |
| the command line | `sbmlsim.serve.cli`, `python -m sbmlsim.serve` |
