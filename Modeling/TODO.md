# TODO

    - what can i learn from one state and infer in another? how to improve?
    - nowcast real future to check if we are not feature leaking
    - simplify layers
    - modify reg data, original/uniform 30 days/7 days fixed
    - try other cutoff regimes, generalize for time window size

    - ensure time sheet reporting
    - improve reconciliation:
        - check concistency for each agg. bin
        - reporting non-0
        - check quality of results ve. models
        - rec. of more than 2 levels
        - add reconciliation methods, try how they work for which level combination
    - feature engineering: 
        - which model uses which features?
        - invent new features, e.g. days between sale and registration
        - which columns in *.csv are used at all?
        - add features for selected models

    - generate test data and verify which patterns are correctly predicted
    - introduce cross-optimize method

    - more conceptual questions: 
        - data cleaning: unregistered reported vechicles?, ...
        - analyze for dealers
    - CHECK: census_tract: CENSUS_TRACT (from registration) or SLS_CENSUS_TRACT (from sales, NaN for non-reporting)
    - inform 
        - decision to use 
            - reporting brands: sale date = SLS_OWNSHP_DT (known at this time) 
            - non-reporting brands: sale date = NVI_OWNSHP_DT, incoming registration data date = NVI_EFCTV_START_DT
            - equivalent to:
                - for sale date: NVI_OWNSHP_DT (for non-reporting) and SLS_OWNSHP_DT (for reporting)
                - for incoming registration data date: NVI_EFCTV_START_DT (for non-reporting) 
        - no RAI used, as there is no plausible sales date for RAI
        - removed rows with SLS_COMMERCIAL_FLAG == "Y" or SLS_DEALER_FLAG == "Y" by Tom's comment that these may not be valid sales but end-of-the-month reporting distortions

        - Questions:
            - comment from discussion with Matt: "Feels we need to revisit the service categories with SPGM to better fit into their world"
    
    - transfer to git

    - check input data - are all column values plausible? I had some doubts, but unsure.
    - change iso week bin key from week to year+week code, ensuring multi-year looping (instead of summing week 1 in 2026 and 2025)


