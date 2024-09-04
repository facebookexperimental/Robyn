from dataclasses import dataclass


@dataclass
class TrialsConfig:
    num_trials: int
    num_iterations_per_trial: int
    timeseries_validation: bool
    add_penalty_factor: bool
    rssd_zero_penalty: bool = True  # Added this line with a default value

    def __str__(self) -> str:
        return (
            f"TrialsConfig("
            f"num_trials={self.num_trials}, "
            f"num_iterations_per_trial={self.num_iterations_per_trial}, "
            f"timeseries_validation={self.timeseries_validation}, "
            f"add_penalty_factor={self.add_penalty_factor}, "
            f"rssd_zero_penalty={self.rssd_zero_penalty}"  # Added this line
            f")"
        )

    def update(
        self,
        num_trials: int,
        num_iterations_per_trial: int,
        timeseries_validation: bool,
        add_penalty_factor: bool,
        rssd_zero_penalty: bool,  # Added this parameter
    ) -> None:
        """
        Update the TrialsConfig parameters.

        :param num_trials: The new number of trials.
        :param num_iterations_per_trial: The new number of iterations per trial.
        :param timeseries_validation: Whether to use timeseries validation.
        :param add_penalty_factor: Whether to add a penalty factor.
        :param rssd_zero_penalty: Whether to apply zero penalty in RSSD calculation.
        """
        self.num_trials = num_trials
        self.num_iterations_per_trial = num_iterations_per_trial
        self.timeseries_validation = timeseries_validation
        self.add_penalty_factor = add_penalty_factor
        self.rssd_zero_penalty = rssd_zero_penalty  # Added this line


# Example usage:
if __name__ == "__main__":
    # Initialize TrialsConfig
    trials_config: TrialsConfig = TrialsConfig(
        num_trials=100,
        num_iterations_per_trial=1000,
        timeseries_validation=True,
        add_penalty_factor=False,
        rssd_zero_penalty=True,  # Added this line
    )

    # Print the TrialsConfig object
    print(trials_config)

    # Update the TrialsConfig
    trials_config.update(
        num_trials=200,
        num_iterations_per_trial=1500,
        timeseries_validation=False,
        add_penalty_factor=True,
        rssd_zero_penalty=False,  # Added this line
    )

    # Print the updated TrialsConfig object
    print(trials_config)
