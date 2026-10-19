
import json
import simpy
from build_model import build_model

if __name__ == "__main__":
    env = simpy.Environment()

    with open("config/model.json") as f:
        config = json.load(f)

    system = build_model(env, config)
    print("System built:", system)

    # Start simulation
    env.run(until=30)
    print("Simulation completed.")
