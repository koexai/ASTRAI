"""
this module is meant to save a snapshot of the training code and start the training
"""
import datetime
from train import main
from log_experiments import save_code

if __name__ == "__main__":
    time_st = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_code("../code_" + time_st + ".zip")
    main(time_st)