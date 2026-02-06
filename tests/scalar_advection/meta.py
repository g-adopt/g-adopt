steps = {
    "non-adaptive": {
        "entrypoint": "scalar_advection.py",
        "outputs": ["final_error.log"],
    },
    "adaptive": {
        "entrypoint": "scalar_advection_adaptive.py",
        "outputs": ["final_error_adaptive.log", "num_steps_adaptive.log", "dt_stats_adaptive.log"],
    },
}

pytest = "local"
