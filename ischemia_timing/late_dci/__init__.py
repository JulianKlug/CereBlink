"""Late-onset DCI analysis (revised analysis plan B).

Layers, each calling only the one below:

    run_analysis     CLI: runs the plan, writes tables / figures
         |
    analyses         plan-level analyses (primary, sensitivity, model check)
         |
    cohort           patient table, landmark and piecewise datasets
    competing_risks  Aalen-Johansen, Fine-Gray, absolute risk, Brier, C-index
         |
    data_sources     file names, decryption, raw sheets
"""
