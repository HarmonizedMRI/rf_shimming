# rf_shimming
RF shimming using GRE and diffusion imaging with EPI readout

1. script_epi_1mm_terra_R3_1shot_rf_shim.m -> generates a diffusion EPI sequence at R=3 with either anti-CP, CP or gradient mode.

2. script_writeGRE_rfShim_Tx_1chan_only.m -> GRE sequence where only a single channel is used for transmission each time.

3. script_writeGRE_rfShim_antiCP_CP_tiamo.m -> GRE sequence with transmission from all 8 channels, and each line of k-space is acquired 4 times (1. dummy, 2. anti-CP, 3. CP and 4. gradient mode).
   
