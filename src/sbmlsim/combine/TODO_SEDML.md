
- [ ] RepeatedTasks with new features
- [ ] 3D plotting
- [ ] parameter fitting


-----------------------------

sbmlsim.Experiment
- globally unique identifiers (validate on object creation)
- identifiers must be SIDs! (so that experiements can be )

- [ ] applied dimensions;


# Documentation
- [ ] create documentation of SED-ML features

- [ ] parse KISAO ontology

# Test cases
- [ ] execute/update all examples for the specification

# Run COMBINE archive
- [x] execute COMBINE archives (refactoring required; also store results)

# Serialization (SimulationExperiment -> SED-ML)
- [ ] datasets !
  - [ ] slices
  - [ ] dataRange
- [x] model
    - [x] resolve model from URN
    - [S] AddXML, ChangeXML, RemoveXML
    - [S] ComputeChange
    - [~S] support amount, concentration and native species targets
- [x] simulation
    - [x] UniformTimecourse
    - [S] OneStep
    - [S] SteadyState
- [~] tasks
    - [x] Task
    - [ ] RepeatedTask
- [x] data generators
- [x] figures
  - [x] plot2d
  - [ ] plot3d 
  - [x] styles
- [ ] reports
- [ ] concentrations/amounts xpath (parse symbols and use for evaluation)
  - [ ] data generators
  - [ ] model changes; changesets  

# Parsing (SED-ML -> SimulationExperiment)
- [ ] datasets !

- [ ] parameters in computation

## Plot2D
- [x] fix reverse axis
- [x] support height and width;
      plots have height & width; these can be directly set in single plot figures;
      for multi-plot figures the combined height and width must be calculated
- [x] use curve.name as label in legend
- [x] support style on axis
- [x] support setting label (name of axis)
- [x] plot width and height  
- [x] curve type (points, bar, ...)
- [~] error bars; assymetrical error bars; see https://github.com/SED-ML/sed-ml/issues/137
- [x] styling of bar plots; 
- [x] yAxis right
- [x] order of abstract curve
- [x] shaded areas & fills
- [x] resolve and apply basestyle
- [x] update bar styling (see https://github.com/SED-ML/sed-ml/issues/140)
- [ ] support figure caption via notes (<notes><p xmlns="xhtml">Figure 1 - Example for figure with text legend and sub-plots.</p></notes>)


# Reports
- [ ] reports
# Repeated Tasks
- [ ] repeated task
# Simulation
- [ ] steady state & one-step
# Kisao terms

## Plot3D
- [ ] Plot3D examples (surfaces)
- [ ] use surface.name as label in legend

# Parameter fitting

# Altair serialization
