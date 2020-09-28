from sbmlsim.test.data.diff import simulate_examples


def test_run_simulations():
    simulate_examples.run_simulations(create_files=False)


def test_run_comparisons():
    simulate_examples.run_comparisons(create_files=False)
