-- SQL Queries for GOOD Grouping Examples (Records = Weeks)
-- These groups have exactly 1 record per week as intended
-- ======================================================================


-- ======================================================================
-- GOOD EXAMPLE 1
-- ======================================================================

-- 1 weeks, 1 records
-- First: 2025-38, Last: 2025-38
SELECT *
FROM [SPGM_Live].[SPGM_Weekly_INV_NVI_SLS_20251107]
WHERE [TRIM_DESCRIPTION] = 'S'
  AND [MAKE_DESC] = 'NISSAN'
  AND [MODEL_DESC] = 'FRONTIER /XE'
  AND [SERIES_TEXT] = 'MTL'
  AND [FUEL_DESC] = 'GASOLINE'
  AND [SEGMENT_DESC] = 'MIDSIZE PICKUP'
  AND [ADVNC_VEH_TYPE_DESC] = 'Gasoline'
  AND [VEH_MDL_YR] = '2025'
  AND [MODEL_KEY_SHORT] = '149631'
  AND [INV_MIN_VIN_SOLD_DT] = 2025-09-25 00:00:00
  AND [INV_LST_PRCE_AMT] = 37014.0
  AND [INV_MSRP_AMT] = 38440.0
  AND [INV_CENSUS_TRACT] = '08013000132'
  AND [INV_CONTROL_NBR] = '0238972'
  AND [INV_DEALER_NAME] = 'VALLEY NISSAN LLC'
  AND [INV_TOWN_NAME] = 'LONGMONT'
  AND [INV_STATE_ABBRV] = 'CO'
  AND [INV_COUNT] = 1.0
ORDER BY [REPORT_YEAR_WEEK];


-- ======================================================================
-- GOOD EXAMPLE 2
-- ======================================================================

-- 1 weeks, 1 records
-- First: 2025-42, Last: 2025-42
SELECT *
FROM [SPGM_Live].[SPGM_Weekly_INV_NVI_SLS_20251107]
WHERE [TRIM_DESCRIPTION] = 'TRAIL BOSS'
  AND [MAKE_DESC] = 'CHEVROLET'
  AND [MODEL_DESC] = 'COLORADO'
  AND [SERIES_TEXT] = 'TB'
  AND [FUEL_DESC] = 'GASOLINE'
  AND [SEGMENT_DESC] = 'MIDSIZE PICKUP'
  AND [ADVNC_VEH_TYPE_DESC] = 'Gasoline'
  AND [VEH_MDL_YR] = '2026'
  AND [MODEL_KEY_SHORT] = '154470'
  AND [INV_MIN_VIN_SOLD_DT] = 1899-12-30 00:00:00
  AND [INV_LST_PRCE_AMT] = 45185.0
  AND [INV_MSRP_AMT] = 45185.0
  AND [INV_CENSUS_TRACT] = '48113000142'
  AND [INV_CONTROL_NBR] = '0024685'
  AND [INV_DEALER_NAME] = 'CLAY COOLEY CHEVROLET'
  AND [INV_TOWN_NAME] = 'IRVING'
  AND [INV_STATE_ABBRV] = 'TX'
  AND [INV_COUNT] = 1.0
ORDER BY [REPORT_YEAR_WEEK];


-- ======================================================================
-- GOOD EXAMPLE 3
-- ======================================================================

-- 1 weeks, 1 records
-- First: 2025-36, Last: 2025-36
SELECT *
FROM [SPGM_Live].[SPGM_Weekly_INV_NVI_SLS_20251107]
WHERE [TRIM_DESCRIPTION] = 'BIG BEND'
  AND [MAKE_DESC] = 'FORD'
  AND [MODEL_DESC] = 'BRONCO SPORT'
  AND [SERIES_TEXT] = 'BGB'
  AND [FUEL_DESC] = 'GASOLINE'
  AND [SEGMENT_DESC] = '*SUBCOMPACT SUV'
  AND [ADVNC_VEH_TYPE_DESC] = 'Gasoline'
  AND [VEH_MDL_YR] = '2025'
  AND [MODEL_KEY_SHORT] = '149608'
  AND [INV_MIN_VIN_SOLD_DT] = 2025-06-13 00:00:00
  AND [INV_LST_PRCE_AMT] = 33833.0
  AND [INV_MSRP_AMT] = 37595.0
  AND [INV_CENSUS_TRACT] = '01121000119'
  AND [INV_CONTROL_NBR] = '0013292'
  AND [INV_DEALER_NAME] = 'TONY SERRA FORD INC'
  AND [INV_TOWN_NAME] = 'SYLACAUGA'
  AND [INV_STATE_ABBRV] = 'AL'
  AND [INV_COUNT] = 1.0
ORDER BY [REPORT_YEAR_WEEK];


-- ======================================================================
-- GOOD EXAMPLE 4
-- ======================================================================

-- 1 weeks, 1 records
-- First: 2025-37, Last: 2025-37
SELECT *
FROM [SPGM_Live].[SPGM_Weekly_INV_NVI_SLS_20251107]
WHERE [TRIM_DESCRIPTION] = 'BIG BEND'
  AND [MAKE_DESC] = 'FORD'
  AND [MODEL_DESC] = 'BRONCO SPORT'
  AND [SERIES_TEXT] = 'BGB'
  AND [FUEL_DESC] = 'GASOLINE'
  AND [SEGMENT_DESC] = '*SUBCOMPACT SUV'
  AND [ADVNC_VEH_TYPE_DESC] = 'Gasoline'
  AND [VEH_MDL_YR] = '2025'
  AND [MODEL_KEY_SHORT] = '149608'
  AND [INV_MIN_VIN_SOLD_DT] = 2025-07-21 00:00:00
  AND [INV_LST_PRCE_AMT] = 34070.0
  AND [INV_MSRP_AMT] = 37255.0
  AND [INV_CENSUS_TRACT] = '01001000208'
  AND [INV_CONTROL_NBR] = '0022077'
  AND [INV_DEALER_NAME] = 'LONG LEWIS OF THE RIVER REGION'
  AND [INV_TOWN_NAME] = 'PRATTVILLE'
  AND [INV_STATE_ABBRV] = 'AL'
  AND [INV_COUNT] = 1.0
ORDER BY [REPORT_YEAR_WEEK];


-- ======================================================================
-- GOOD EXAMPLE 5
-- ======================================================================

-- 7 weeks, 7 records
-- First: 2025-36, Last: 2025-42
SELECT *
FROM [SPGM_Live].[SPGM_Weekly_INV_NVI_SLS_20251107]
WHERE [TRIM_DESCRIPTION] = '250 4MATIC'
  AND [MAKE_DESC] = 'MERCEDES-BENZ'
  AND [MODEL_DESC] = 'GLB'
  AND [SERIES_TEXT] = '250'
  AND [FUEL_DESC] = 'GASOLINE'
  AND [SEGMENT_DESC] = 'NEAR LUXURY SUV'
  AND [ADVNC_VEH_TYPE_DESC] = 'Mild Hybrid Electric Vehicle Gasoline'
  AND [VEH_MDL_YR] = '2026'
  AND [MODEL_KEY_SHORT] = '154514'
  AND [INV_MIN_VIN_SOLD_DT] = 1899-12-30 00:00:00
  AND [INV_LST_PRCE_AMT] = 51455.0
  AND [INV_MSRP_AMT] = 51455.0
  AND [INV_CENSUS_TRACT] = '34007006033'
  AND [INV_CONTROL_NBR] = ''
  AND [INV_DEALER_NAME] = ''
  AND [INV_TOWN_NAME] = ''
  AND [INV_STATE_ABBRV] = ''
  AND [INV_COUNT] = 2.0
ORDER BY [REPORT_YEAR_WEEK];

