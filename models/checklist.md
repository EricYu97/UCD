1. Check if the model can be used for multi-channel inputs
2. Check if the model can be used for multi-class outputs
3. Check if the model contains sigmoid at the end, if so the model can only be used with BCE loss.
4. Ensure that the model outputs are integrated into a dict : {"main_predictions": *, "aux_predictions": *,*,...}

DASNet implemented not correct.
AERNet outputs NAN and cannot be trained.
USSFCNet the loss convergence is really slow.
HANet, the loss is NAN
Tiny_CD cannot convergence
DSIFN not convergence