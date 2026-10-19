import unittest
from dataclasses import replace
from agenticsim.contracts import Quantity, Request, require_capabilities
from agenticsim.demo import CoolingBackend, calibrate_grid, experiment


class ContractTests(unittest.TestCase):
    def test_recovers_synthetic_rate(self):
        result = experiment()["calibration"]
        self.assertEqual(result["best"]["parameter"]["value"], 0.12)
        self.assertEqual(result["best"]["mse"], 0)
        self.assertEqual(len(result["evaluations"]), 30)

    def test_rejects_units_names_and_invalid_values(self):
        for values in ({}, {"rate": Quantity(0.1, "Hz")}, {"rate": Quantity(float("nan"), "1/s")},
                       {"rate": Quantity(-1, "1/s")}, {"rate": Quantity(2, "1/s")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CoolingBackend().simulate(Request(values, (0, 1)))

    def test_rejects_time_errors(self):
        for times in ((), (1, 0), (0, 0), (-1, 0), (float("inf"),)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                CoolingBackend().simulate(Request({"rate": Quantity(.1, "1/s")}, times))

    def test_no_invented_gradient(self):
        with self.assertRaises(ValueError):
            require_capabilities(CoolingBackend.card, {"pathwise_gradient"})

    def test_observation_contract(self):
        backend = CoolingBackend()
        observed = backend.simulate(Request({"rate": Quantity(.1, "1/s")}, (0, 1)))
        for invalid in (replace(observed, unit="degC"), replace(observed, values=(1,)),
                        replace(observed, observable_id="other"), replace(observed, values=(1, float("nan")))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                calibrate_grid(backend, invalid, "rate", (Quantity(.1, "1/s"),))


if __name__ == "__main__":
    unittest.main()
