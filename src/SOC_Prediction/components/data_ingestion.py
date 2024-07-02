import os
import pybamm
import numpy as np
import pandas as pd
from SOC_Prediction import logger
from SOC_Prediction.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
        
    def term_charge_voltage(self, name):
        match name:
            case "Ai2020": return "4.0"
            case "Chen2020": return "4.05"
            case "Marquis2019": return "3.85"
            case _: return "4.00"

    def generate_data(self)-> str:
        '''
        Fetch data from the pybamm model
        '''

        try: 
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            rng = np.random.default_rng(1000)
            param_set = ["Marquis2019","Ai2020","Chen2020"]
            for chemistry in param_set:
                # ===================================================================
                # Model and Parameters
                # ===================================================================
                model = pybamm.lithium_ion.DFN()
                params = pybamm.ParameterValues(chemistry)



                CV = self.term_charge_voltage(chemistry)
                tdrive1 = np.arange(7200)
                cdrive1 = 1 * rng.random(7200)
                drive_cycle1 = np.column_stack([tdrive1, cdrive1])

                tdrive2 = np.arange(3600)
                cdrive2 = 2 * rng.random(3600)
                drive_cycle2 = np.column_stack([tdrive2, cdrive2])

                tdrive3 = np.arange(1800)
                cdrive3 = 4 * rng.random(1800)
                drive_cycle3 = np.column_stack([tdrive3, cdrive3])
                T = [10,25,35,-5]

                for ta in T:
                    params["Ambient temperature [K]"] = ta + 273.15
                    nom_cap = params["Nominal cell capacity [A.h]"]


                    exp = pybamm.Experiment(["Rest for 10 minutes",
                                        pybamm.step.c_rate(drive_cycle1, termination="3.0V"),
                                        "Rest for 20 minutes",
                                        ("Charge at 2C until " + CV + " V"),
                                        ("Hold at " + CV + " V until C/50"),
                                        "Rest for 20 minutes",
                                        pybamm.step.c_rate(drive_cycle2, termination="3.0V"),
                                        "Rest for 20 minutes",
                                        ("Charge at 2C until " + CV + " V"),
                                        ("Hold at " + CV + " V until C/50"),
                                        "Rest for 20 minutes",
                                        pybamm.step.c_rate(drive_cycle3, termination="3.0V"),
                                        "Rest for 20 minutes",
                                        ("Charge at 2C until " + CV + " V"),
                                        ("Hold at " + CV + " V until C/50"),
                                        "Rest for 20 minutes"]*self.config.ncycles_param,
                                        period="1 seconds")

                    # ===================================================================
                    # Simulation
                    # ===================================================================
                    sim = pybamm.Simulation(model, experiment=exp, parameter_values=params)
                    sim.solve(initial_soc=0.8)
                    # ===================================================================
                    # Access Variables
                    # ===================================================================
                    sol = sim.solution
                    t = sol["Time [s]"].entries
                    i = sol["C-rate"].entries
                    v = sol["Voltage [V]"].entries
                    Ta = sol["Ambient temperature [C]"].entries
                    s0 = nom_cap * 0.8
                    s = (s0 - sol["Discharge capacity [A.h]"].entries)/s0

                    # ===================================================================
                    # Save results to csv file
                    # ===================================================================
                    t = t.reshape((t.size, 1))
                    i = i.reshape((i.size, 1))
                    v = v.reshape((v.size, 1))
                    Ta = Ta.reshape((Ta.size, 1))
                    s = s.reshape((s.size, 1))
                    f = np.concatenate((t, i, v, Ta, s), 1)
                    header = "time, c_rate, v, a_temp, soc"
                    np.savetxt(f"{self.config.data_dir}/{chemistry}_rand_{abs(ta)}.csv", f, delimiter=',', header=header)
                data1 = pd.read_csv(f"{self.config.data_dir}/{chemistry}_rand_{25}.csv")
                data2 = pd.read_csv(f"{self.config.data_dir}/{chemistry}_rand_{10}.csv")
                data3 = pd.read_csv(f"{self.config.data_dir}/{chemistry}_rand_{35}.csv")
                data4 = pd.read_csv(f"{self.config.data_dir}/{chemistry}_rand_{5}.csv")



                data_merged = pd.concat([data1,data2,data3,data4],ignore_index=True)
                data_merged.to_csv(f"{self.config.data_dir}/{chemistry}_rand_{abs(ta)}.csv",index = False)
                logger.info(f"Created data for {chemistry}")

        except Exception as e:
            raise e