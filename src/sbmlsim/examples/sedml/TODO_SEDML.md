# Run COMBINE archive
- [x] execute COMBINE archives (refactoring required; also store results)

# Serialization (SimulationExperiment -> SED-ML)
- [x] model
    - [x] resolve model from URN
    - [S] AddXML, ChangeXML, RemoveXML
    - [S] ComputeChange
- [x] simulation
    - [x] UniformTimecourse
    - [S] OneStep
    - [S] SteadyState
    


# Plotting


# Parsing (SED-ML -> SimulationExperiment)
## Plot2D
- [x] support height and width;
      plots have height & width; these can be directly set in single plot figures;
      for multi-plot figures the combined height and width must be calculated
- [x] use curve.name as label in legend
- [x] support style on axis
- [~] support setting label (name of axis), see https://github.com/fbergmann/libSEDML/issues/110
- [x] plot width and height  
- [x] curve type (points, bar, ...)
  - [x] fix stacking position
  - [~] bars next to each other; see https://github.com/SED-ML/sed-ml/issues/138  
- [~] error bars; see https://github.com/SED-ML/sed-ml/issues/137
- [x] styling of bar plots; 
- [x] yAxis right
- [x] order of abstract curve
- [x] shaded areas & fills
- [x] resolve and apply basestyle

## Altair serialization

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
