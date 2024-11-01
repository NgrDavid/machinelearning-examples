import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from main import HiddenMarkovModel

if __name__ == "__main__":

    print("Loading tracking data.")
    data_path = "./data/TrackingData.csv"
    df = pd.read_csv(data_path)

    # number of hidden discrete states that the model will use. We are choosing 3 for demo purposes.
    num_states = 3
    # dimensionality of the data. Since we are using velocity and acceleration, dimensionality is 2.
    dimensions = 2
    # default type of observations model. See https://github.com/lindermanlab/ssm/blob/6c856ad3967941d176eb348bcd490cfaaa08ba60/ssm/hmm.py#L81 for more.
    observations_model_type = "gaussian"
    # default type of transitions model. See https://github.com/lindermanlab/ssm/blob/6c856ad3967941d176eb348bcd490cfaaa08ba60/ssm/hmm.py#L51 for more.
    transitions_model_type = "stationary"

    print(f"Initializing HMM.")
    hmm = HiddenMarkovModel(num_states=num_states, dimensions=dimensions,
                            observations_model_type=observations_model_type, transitions_model_type=transitions_model_type)
    
    # variables to estimate.
    vars_to_estimate = {
        "initial_state_distribution": False,
        "transitions_params": True,
        "observations_params": True
    }
    # batch size.
    batch_size = len(df)
    # max number of training iterations.
    max_iter = 100
    
    print("Training HMM.")
    for observation in zip(df["Velocity"].to_list(), df["Acceleration"].to_list()):
        hmm.fit_async(observation=observation, vars_to_estimate=vars_to_estimate,
                    batch_size=batch_size, max_iter=max_iter)

    while True:
        if hmm.get_fit_finished():
            hmm.reset_fit_loop()
            break
        time.sleep(1)

    print("Saving model state to file.")
    hmm.save_model("./data/hmm_model.pkl")