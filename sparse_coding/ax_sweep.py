import time
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax import ComparisonOp, Metric, OutcomeConstraint
from sparse_coding.train_autoencoder import run_trial

def ax_sweep():
    ax_client = AxClient()
    ax_client.create_experiment(
        name="sae_regularization",
        parameters=[
            {
                "name": "reg",
                "type": "string",
                "values": ["l1", "sqrt", "combined_hoyer_sqrt"]
            },
            {
                "name": "l1_coeff",
                "type": "range",
                "bounds": [0.00001, 0.001],
            },
            {
                "name": "reg_coeff_2",
                "type": "range",
                "bounds": [0.00001, 0.001],
            },
        ],
        objectives={"reconstruction_loss": ObjectiveProperties(minimize=True)},
        outcome_constraints=[
            OutcomeConstraint(metric=Metric(name="l0"), op = ComparisonOp.GEQ, bound=29),
            OutcomeConstraint(metric=Metric(name="l0"), op = ComparisonOp.LEQ, bound=31)
        ],
    )

    def evaluate(parameters):
        reconstruction_loss, l0 = run_trial(parameters)
        return {
            "reconstruction_loss": reconstruction_loss,
            "l0": l0
        }

    num_trials = 0
    total_trials_we_will_run = 25

    timestamp = time.time()

    while num_trials <= total_trials_we_will_run:
        trial_idx_to_parameters, is_experiment_complete = ax_client.get_next_trials(
            max_trials=1
        )
        num_trials += len(trial_idx_to_parameters)

        for trial_index, parameters in trial_idx_to_parameters.items():
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=evaluate(parameters)
            )

        # save after each trial in case it crashes partway through
        df = ax_client.get_trials_data_frame()
        df.to_csv(f'/workspace/ax_sweep_{timestamp}')

    print(df)

if __name__ == "__main__":
    ax_sweep()
