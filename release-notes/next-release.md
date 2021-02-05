# Release notes for sbmlsim x.y.z

- Fix #81, feature timecourse concatenation and reuse in timecoursesim

This allows simplified creation of multiple dosing timecourse simulations and reuse
of timecourse definitions in multiple simulations. E.g. repeated timecourse `[tc]*3`

``` 
    simulator = Simulator(MODEL_REPRESSILATOR)
    tc = Timecourse(
                    start=0,
                    end=50,
                    steps=100,
                    changes={"X": 10})

    s = simulator._timecourse(
        simulation=TimecourseSim(
            [tc]*3
        )
    )
```

- Fix #82, bugfix; remove time shift from discarded simulations
