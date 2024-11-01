# Summary

This example shows how to fit an HMM model using previously collected data and then load the trained model in Bonsai to run inference online.

# Instructions

You need to create a python environment with the correct packages. I suggest using a virtual environment to keep all of the package dependencies isolated. You can just copy and paste the following.

```python
python -m venv .venv
.\.venv\Scripts\activate
pip install numpy cython ipykernel pandas matplotlib
pip install --no-build-isolation ssm@git+https://github.com/lindermanlab/ssm@6c856ad3967941d176eb348bcd490cfaaa08ba60
python .\python\fit_hmm.py
```

If everything runs correctly, this should generate a file called `data\hmm_model.pkl`. This file is what Bonsai will use to load the model.

Similar to the python virtual environment, I recommend creating a dedicated Bonsai environment. The zip file should already contain the required `Bonsai.config` and `NuGet.config` files for specifying the packages. To do this, you need to have already installed the dotnet-sdk which you can find [here](https://dotnet.microsoft.com/en-us/download). Then, you run the following:

```cmd
dotnet new install Bonsai.Templates::2.8.4
dotnet new bonsaienv
```

When prompted to run the Setup.ps1 script, enter yes. You should start to see packages being installed into your `.bonsai` folder. Finally. run:

```cmd
.\.bonsai\Bonsai.exe
```

This will launch Bonsai, where you can select the `workflows\HMM_online_inference.bonsai` file and run. It is currently just reading off values from the tracking data file but the `TrackingData` node can be changed to any data source. You can visualize the state probabilities by opening the `HMMStateProbabilities` node.