# from re import T  # unused import, breaks Python 3.13
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "3000000"

print("\n\nrestarting.")
if 1: #imports, #in local setup assumes "python: Select Interpreter" -> "Python 3.11.7 ('base')" refering to  ~/anaconda3/bin/python
    import pandas as pd
    import numpy as np
    import os
    from sqlalchemy import create_engine
    import time
    import pdb
    from collections import Counter
    import copy
    import re
    import math
    import matplotlib.pyplot as plt

if 1: #helpers
    class Timer:
        """
        Simple timer to measure elapsed time.
        Usage: timer = Timer(); do_someting(); elapsed = timer.time()
        """
        def __init__(self):
            self.start = time.time()
        
        def time(self):
            return time.time() - self.start

    def print_full_df(df, max_colwidth=15): #15 is good not to print e.g. time in datetime columns
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', None,
            'display.max_colwidth', max_colwidth,
            'display.expand_frame_repr', False
        ):
            print(df)

if 1: # Create SQLAlchemy engine
    connection_string = (
        f"mssql+pyodbc://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}"
        f"@10.0.30.16:1433/RTID_SourceData"
        f"?driver=ODBC+Driver+17+for+SQL+Server"
        f"&TrustServerCertificate=yes"
        f"&Connection+Timeout=5"
        )
    engine = create_engine(connection_string)

### data preparation ########################################################################################################################################

db_name4 = "RTID_SourceData.SPGM_Live.SPGM_Weekly_INV_NVI_SLS_20260209"
verb = 2 #1: few lines per df, 2: full details, create lookup info for quick info

if 1: #import reference data: census tracts, model keys 
    def load_df(source_filename, remove_columns=None):
        """Load dataframe from CSV cache or original source file."""
        csv_filename = source_filename + ".csv"
        
        if os.path.exists(csv_filename):
            return pd.read_csv(csv_filename)
        else:
            #check if source_filename is an excel file  
            if source_filename.endswith('.xlsx'):
                df = pd.read_excel(source_filename)
            else:
                df = pd.read_csv(source_filename)
            
            if remove_columns is not None:
                df = df.drop(columns=remove_columns)
            
            df.to_csv(csv_filename, index=False)
            return df


    def load_CT():
        """Load census tract data from CSV or Excel, removing unnecessary columns."""
        excel_filename = "List of US Census Tract Centroids_20251003.xlsx"        
        """ has columns: 
            ['FIPsCode', 'CensusTract', 'StateCode', 'State_Name', 'StateAbbrv',
            'CountyCode', 'CountyName', 'LandArea', 'WaterArea', 'Pop2020',
            'PopCentroid_Lat', 'PopCentroid_Lon', 'GeometricCentroid_Lat',
            'GeometricCentroid_Lon'], """
        cols_to_remove = ['State_Name', 'CountyCode', 'CountyName', 'LandArea', 'WaterArea', 'GeometricCentroid_Lat', 'GeometricCentroid_Lon']       
        df = load_df(excel_filename, cols_to_remove)        
        #prepare dict for fast lookup
        ct_lat_lon = dict(zip(df['FIPsCode'], 
                        zip(df['PopCentroid_Lat'], 
                            df['PopCentroid_Lon'])))

        return df, ct_lat_lon

    ct_df, ct_lat_lon = load_CT()

    def distance_between_ct(ct1, ct2):
        """Return the great-circle distance between two census tracts (in format with 2 decimals) in miles."""

        def latlondist(lat1, lon1, lat2, lon2):
            """Return the great-circle distance between two lat/long points in miles."""
            '''lat: equator =0, north pole = 90, south pole = -90'''
            '''lon: 0 = greenwich, going east = positive, going west = negative, 180 = antimeridian'''
            radius = 3958.8
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            delta_phi = phi2 - phi1
            delta_lambda = math.radians(lon2 - lon1)
            a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
            return 2 * radius * math.asin(math.sqrt(a))

        ct1=round(ct1*100); lat1, lon1 = ct_lat_lon[ct1]
        ct2=round(ct2*100); lat2, lon2 = ct_lat_lon[ct2]
        return latlondist(lat1, lon1, lat2, lon2)

    model_short_key_df = load_df("HYU_MODEL_KEY_SHORT_5YR_202506.txt")

    grouped = model_short_key_df.groupby(['MAKE_DESC', 'MODEL_DESC', 'ADVNC_VEH_TYPE_DESC'])['MODEL_KEY_SHORT'].apply(list)
    key_dict = {k: v for v in grouped for k in v}

us_states=["CO", "SC", "MN", "OK", "NM"]
state_list = "'"+"','".join(us_states)+"'"
if 1: #define queries for dfs
    queries = {}
    queries["us5b_df"]= f"""SELECT * FROM {db_name4} WHERE SLS_STATE_ABBRV IN ({state_list}) OR INV_STATE_ABBRV IN ({state_list}) OR NVI_STATE_ABBRV IN ({state_list})"""
    df_names=queries.keys() 

dfs = {}
for df_name in ["us5b_df"]: 
    print("="*150+f"\nProcessing df {df_name}")

    if 1: # Load dataframes
        def load_df_from_db_or_csv(engine, query, name):
            """
            Load DataFrame from CSV if it exists, otherwise load from database and save to CSV.
            
            Parameters:
            - engine: Database connection engine
            - query: SQL query to execute if CSV doesn't exist
            - csv_name: Name of the CSV file to check/save
            
            Returns:
            - DataFrame loaded from CSV or database
            """
            csv_name=name+".csv"
            if os.path.exists(csv_name):
                df = pd.read_csv(csv_name)
                print(f"Loaded {name}: {len(df)} rows (from file)")
            else:
                start_time = time.time()
                df = pd.read_sql(query, engine)
                elapsed = time.time() - start_time
                df.to_csv(csv_name, index=False)
                print(f"Loaded {name}: {len(df)} rows (from database in {elapsed:.2f}s)")
            
            df.attrs['name'] = name #typical convention for storing metadata in pandas dataframe
            return df
        dfs[df_name] = load_df_from_db_or_csv(engine, queries[df_name], df_name)

    if 1: #remove columns that have only one unique value (including NaN)
        def remove_single_value_columns(df):
            """
            Identifies and removes columns that contain only one unique value.
            Prints the columns and their single value before removal.
            
            Parameters:
            -----------
            df : pandas.DataFrame
                The input dataframe to process (must have df.attrs['name'] set)
                
            Returns:
            --------
            pandas.DataFrame
                Reduced dataframe with single-value columns removed
            """
            single_value_cols = {}
            
            # Identify columns with only one unique value
            for col in df.columns:
                # Count unique values including NaN/None
                n_unique = df[col].nunique(dropna=False)
                
                if n_unique == 1:
                    # Get the single unique value
                    unique_val = df[col].iloc[0]
                    single_value_cols[col] = unique_val
            
            # Print the dictionary
            name = df.attrs.get('name', 'unnamed_df')
            print(f"{name}: removing single-value columns {single_value_cols}")
            
            # Remove these columns and return reduced dataframe
            df_reduced = df.drop(columns=list(single_value_cols.keys()))
            
            return df_reduced
        dfs[df_name] = remove_single_value_columns(dfs[df_name])

    if 1: #check for remaining NaN content
        def print_nan_columns(df):
            """Print dict of columns with their NaN counts."""
            name = df.attrs.get('name', 'unnamed_df')
            nan_dict = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}
            if nan_dict:
                print(f"Columns with NaN values in df {name}: {nan_dict}")
        print_nan_columns(dfs[df_name])

    if 1: #print 5 top values per df
        def print_top_values(df, top_n=5, sort=None, add_dtype=False):
            name = df.attrs.get('name', 'unnamed_df')
            print(f"\ntop {top_n} values for df {name}:")
            
            # Determine column order based on sort parameter
            if sort == "alpha":
                cols = sorted(df.columns)
            elif sort == "unique":
                cols = sorted(df.columns, key=lambda col: df[col].nunique(dropna=False))
            else:  # None - use original order
                cols = df.columns
            
            for col in cols:
                col_data = df[col].replace('', '__EMPTY__')
                value_counts = col_data.value_counts(dropna=False).head(top_n)
                n_unique = col_data.nunique(dropna=False)
                
                values_str = ", ".join([
                    f"{val if val != '__EMPTY__' else '(empty)'} ({count})" 
                    for val, count in value_counts.items()
                ])
                
                dtype_str = f" {df[col].dtype}" if add_dtype else ""
                print(f"{col} ({n_unique}): {values_str}{dtype_str}")
        if verb>1: #print top occuring values of reg_df and rai_df
            print_top_values(dfs[df_name], add_dtype=True)
            
    if 1: #remove single-value columns, duplicate rows
        print("removing duplicate columns, rows...")
        def remove_duplicates_inplace(df):
            """Keep only the first row for duplicate rows and return the reduced df."""
            name = df.attrs.get('name', 'unnamed_df')
            len_before=len(df)
            df.drop_duplicates(inplace=True)
            len_after=len(df)
            if len_before-len_after > 0:
                print(f"removed {len_before-len_after} duplicates from {name}")
            return df
        remove_duplicates_inplace(dfs[df_name]) #par_df: 6063 removed
    
    if 0: #in-place uppercasing of text columns
        print("uppercasing...")
        def text_col_upper(df):
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.upper() #in-place uppercasing
        text_col_upper(dfs[df_name])



    ### analyze datetime data ########################################################################################################################################

    if 1: #convert datetime strings to datetime type
        def convert_datetime_strings(df):
            def identify_date_string_columns(df):
                """
                Identify columns that contain date/datetime strings in common formats.
                
                Parameters:
                -----------
                df : pandas.DataFrame
                    The dataframe to analyze
                
                Returns:
                --------
                list
                    List of column names that contain date/datetime strings
                """
                # Patterns for different date/datetime formats
                date_pattern = r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD
                datetime_pattern = r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$'  # YYYY-MM-DD HH:MM:SS
                
                string_date_cols = []
                
                for col in df.select_dtypes(include=['object']).columns:
                    sample = df[col].dropna()
                    if len(sample) > 0:
                        # Convert to string and check if all values match date or datetime patterns
                        sample_str = sample.astype(str)
                        if (sample_str.str.match(date_pattern).all() or 
                            sample_str.str.match(datetime_pattern).all()):
                            string_date_cols.append(col)
                
                return string_date_cols
            string_datetime_cols = identify_date_string_columns(df)
            #print("Date/datetime string columns:", string_datetime_cols)
            #convert to datetime type, pd.to_datetime is smart enough to handle both formats
            df[string_datetime_cols] = df[string_datetime_cols].apply(pd.to_datetime)
            #print_top_values(comb2_df,"comb2_df",sort_alphabetically=False,add_dtype=True)
        convert_datetime_strings(dfs[df_name])

print("done2")


if 1: # plot multiple timelines of incoming vehicle data
    def plot_multiple_timelines(df,
                                date_cols,
                                start_date_str,
                                invalid_date_threshold, #dates before this date ...
                                replacement_date, #... are replaced with this date.
                                filename_prefix,
                                drop_last=True):
        co3_df = df.copy()

        if date_cols is None:
            date_cols = [
                "NVI_EFCTV_START_DT",
                "NVI_SYS_CREATE_DT",
                "NVI_OWNSHP_DT",
                "SLS_EFCTV_START_DT",
                "SLS_OWNSHP_DT",
                "INV_MIN_FIRST_SCRAPED_DT",
                "INV_MAX_LAST_STATUS_DT",
            ]

        replacement_ts = pd.Timestamp(replacement_date)
        invalid_ts = pd.Timestamp(invalid_date_threshold)

        # replace invalid dates
        for col in co3_df.select_dtypes(include=["datetime64[ns]"]).columns:
            co3_df[col] = co3_df[col].mask(co3_df[col] < invalid_ts, replacement_ts)

        # Ensure they are datetime (safe-cast)
        for col in date_cols:
            co3_df[col] = pd.to_datetime(co3_df[col], errors="coerce")

        # Collect all unique dates across columns
        all_dates = (
            pd.concat([co3_df[col] for col in date_cols])
            .dropna()
            .sort_values()
            .unique()
        )

        # Create result table with column 'x' holding all occurring date values
        result = pd.DataFrame({"x": all_dates})

        # For each datetime column, compute cumulative count of entries up to each date x
        for col in date_cols:
            counts = (
                co3_df[col]
                .dropna()
                .value_counts()
                .sort_index()
            )
            cum_counts = counts.cumsum()
            cum_counts_aligned = (
                cum_counts
                .reindex(all_dates, method="ffill")
                .fillna(0)
                .astype(int)
            )
            result[f"count_{col}"] = cum_counts_aligned.values

        if drop_last: #drop last value
            plot_df = result.iloc[:-1].copy()
        else:
            plot_df = result.copy()
        plot_df["x"] = pd.to_datetime(plot_df["x"])

        if start_date_str:
            start_date = pd.to_datetime(start_date_str, format="%y-%m-%d")
            plot_df = plot_df[plot_df["x"] >= start_date].sort_values("x")
        else:
            start_date_str = "[no start date]"

        count_cols = [col for col in plot_df.columns if col.startswith("count_")]

        plt.figure(figsize=(14, 7))
        lines = []
        labels = []
        for col in count_cols:
            line, = plt.plot(plot_df["x"], plot_df[col], label=col)
            lines.append(line)
            labels.append(col)

        legend = plt.legend(lines, labels, title="Cumulative Counts", loc="upper left")
        for text, line in zip(legend.get_texts(), lines):
            text.set_color(line.get_color())

        plt.title(f"Cumulative Counts of Date Columns Over Time (Start >= {start_date_str})")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Count")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        filename = f"{filename_prefix}_start_{start_date_str}.png"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved as {filename}")
    plot_multiple_timelines(dfs["us5b_df"],
                            date_cols=None,
                            start_date_str="24-10-01",
                            invalid_date_threshold="1900-01-01",
                            replacement_date="2024-09-30",
                            filename_prefix="cumulative_counts_us5a",
                            drop_last=False)

def cross_with_totals(df, col1, col2):
    if df.empty:
        return pd.DataFrame({"TOTAL": [0]}, index=["TOTAL"])
    tmp = df.copy()
    tmp[col1] = tmp[col1].fillna("NULL").replace("", "EMPTY")
    tmp[col2] = tmp[col2].fillna("NULL").replace("", "EMPTY")
    pt = pd.crosstab(tmp[col1], tmp[col2], dropna=False)
    pt.loc["TOTAL"] = pt.sum(axis=0)
    pt["TOTAL"] = pt.sum(axis=1)
    return pt #use for e.g. print(pt)
def print_brand_reporting(tmp,top_results=None):
    brand_tbl = (
        tmp.groupby("MAKE_DESC")
        .agg(
            registered=("is_registered", "sum"),
            reported=("is_reported", "sum")
        )
    )

    brand_tbl["reported_pct_of_registered"] = (
        brand_tbl["reported"] / brand_tbl["registered"] * 100
    )
    
    brand_tbl = brand_tbl.sort_values("registered", ascending=False)
    if top_results is not None:
        brand_tbl = brand_tbl.head(top_results)
    print_full_df(brand_tbl)
def add_state_from_ct(df, ct_column, state_column):
    fips = {'01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', 
            '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL', 
            '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', 
            '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', 
            '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', 
            '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', 
            '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', 
            '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', 
            '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', 
            '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', 
            '56': 'WY', '72': 'PR'}
    
    # Replace empty strings with NA for consistent null handling
    ct_clean = df[ct_column].replace(['', ' '], pd.NA)
    
    # Process and map, then fill unmapped/null values with 'ZZ'
    df[state_column] = (ct_clean.astype('Int64').astype(str)
                        .str.zfill(11).str[:2].map(fips)
                        .fillna('ZZ'))
    return df
def add_state_from_ct0(df, ct_column, state_column):
    fips = {'01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', 
            '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL', 
            '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', 
            '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', 
            '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', 
            '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', 
            '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', 
            '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', 
            '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', 
            '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', 
            '56': 'WY', '72': 'PR'}
    df[state_column] = (df[ct_column].astype('Int64').astype(str)
                        .str.zfill(11).str[:2].map(fips))
    return df
def add_county_from_ct(df, ct_column, county_column):
    fips = {'01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
            '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
            '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
            '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
            '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
            '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
            '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
            '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
            '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
            '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
            '56': 'WY', '72': 'PR'}

    ct_clean = df[ct_column].replace(['', ' '], pd.NA)
    ct = ct_clean.astype('Int64').astype(str).str.zfill(11)
    state_fips = ct.str[:2]
    county_fips = ct.str[2:5]

    df[county_column] = (state_fips.map(fips) + county_fips).fillna('-')
    return df

if 1: #filter for clean data first
    print("\n"*5+"="*150)
    tmp = dfs["us5b_df"].copy()

    if 1: #define is_registered and is_reported
        print("starting with",len(tmp),"records")
        tmp["is_registered"] = tmp["NVI_OWNSHP_DT"] > pd.Timestamp("1900-01-01")
        tmp["is_reported"]   = tmp["SLS_OWNSHP_DT"] > pd.Timestamp("1900-01-01")
        #print(cross_with_totals(tmp, "is_registered", "is_reported"))
        if len(tmp[~tmp["is_registered"] & ~tmp["is_reported"]]) > 0:
            print("ERROR: there is an item where is_registered is False and is_reported is False")
        
    if 1: #add state from NVI census tract and county from NVI census tract
        #print([col for col in tmp.columns if "CENSUS_TRACT" in col])
        tmp = add_state_from_ct(tmp, "NVI_CENSUS_TRACT", "NVI_STATE_from_ct")
        #tmp = add_state_from_ct(tmp, "SLS_CENSUS_TRACT", "SLS_STATE_from_ct")
        #tmp = add_state_from_ct(tmp, "INV_CENSUS_TRACT", "INV_STATE_from_ct")
        tmp["CENSUS_TRACT"] = tmp["NVI_CENSUS_TRACT"]

        tmp = add_county_from_ct(tmp, "NVI_CENSUS_TRACT", "COUNTY")
        
    if 0: # remove reported cars that have no dealer state
        len_before = len(tmp)
        tmp = tmp[(~tmp["is_reported"]) | (tmp["SLS_STATE_ABBRV"].notna())]
        print(f"removed {len_before - len(tmp)} rows with no dealer state")

    if 1: #exclude cross-state rows, but allow unspecified state
        orig=len(tmp)
        tmp = tmp[
            ((tmp["INV_STATE_ABBRV"].isna()) & (tmp["SLS_STATE_ABBRV"].isna())) |
            ((tmp["INV_STATE_ABBRV"].isin(us_states)) & (tmp["SLS_STATE_ABBRV"].isna())) |
            ((tmp["INV_STATE_ABBRV"].isna()) & (tmp["SLS_STATE_ABBRV"].isin(us_states))) |
            ((tmp["INV_STATE_ABBRV"].isin(us_states)) & (tmp["INV_STATE_ABBRV"] == tmp["SLS_STATE_ABBRV"]))
        ]
        #convert nan states to XX
        tmp["INV_STATE_ABBRV"] = tmp["INV_STATE_ABBRV"].fillna("XX")
        tmp["SLS_STATE_ABBRV"] = tmp["SLS_STATE_ABBRV"].fillna("XX")
        #select for us_states only with both states in us_states
        #tmp = tmp[(tmp["INV_STATE_ABBRV"] == tmp["SLS_STATE_ABBRV"]) & (tmp["INV_STATE_ABBRV"].isin(us_states))]
        #print(f"removed {orig - len(tmp)} rows with cross-state rows")
        print(cross_with_totals(tmp, "INV_STATE_ABBRV", "SLS_STATE_ABBRV"))

    if 1: #add OWNSHP_DT and derived features to tmp
        tmp['OWNSHP_DT'] = np.where(tmp['is_reported'], tmp['SLS_OWNSHP_DT'], tmp['NVI_OWNSHP_DT']) #is_reported is True: take SLS_OWNSHP_DT, otherwise take NVI_OWNSHP_DT
        tmp["day_of_week"] = tmp["OWNSHP_DT"].dt.weekday #0=Monday, 6=Sunday
        tmp["day_of_month"] = tmp["OWNSHP_DT"].dt.day
        tmp['remaining_day_of_month'] = tmp['OWNSHP_DT'].dt.days_in_month - tmp['OWNSHP_DT'].dt.day + 1 #last day of month = 1

    #tmp["NVI_received_dt"] = tmp["NVI_EFCTV_START_DT"]

    if 1:  # remove brands with less than 0.1% market share, define reporting_brands and non_reporting_brands, add column is_reporting_brand
        #print_brand_reporting(tmp) #BMW, Mercedes, Tesla: 0% reported. as expected. But many @ 90% reported. Where is gap? Analyze: registered and not reported.
        # Print unique values of MAKE_DESC and count
        # print(tmp["MAKE_DESC"].value_counts())
        len_before = len(tmp)
        threshold = int(0.001 * len(tmp))
        valid_makes = tmp["MAKE_DESC"].value_counts()[tmp["MAKE_DESC"].value_counts() > threshold].index
        #print(f"valid_makes: {valid_makes}")
        tmp = tmp[tmp["MAKE_DESC"].isin(valid_makes)]
        print(f"removed {len_before - len(tmp)} rows with brands with less than 0.1% market share")
        #print(tmp["MAKE_DESC"].value_counts())
        reporting_brands = tmp[tmp["is_reported"] == True]["MAKE_DESC"].unique().tolist()
        non_reporting_brands = tmp[~tmp["MAKE_DESC"].isin(reporting_brands)]["MAKE_DESC"].unique().tolist()
        print(f"reporting_brands: {reporting_brands}, non_reporting_brands: {non_reporting_brands}")
        tmp["is_reporting_brand"] = tmp["MAKE_DESC"].isin(reporting_brands)

    print(cross_with_totals(tmp, "is_registered", "is_reported"))

    print("SLS_OWNSHP_DT for non-reported cars:")
    print(tmp[tmp["is_reported"] == False].groupby("SLS_OWNSHP_DT").size().reset_index(name="count"))

    if 1: #remove rows with NVI_STATE_from_ct outside first n US_states
        first_n_states=5
        before=len(tmp)
        tmp=tmp[tmp["NVI_STATE_from_ct"].isin(us_states[:first_n_states])] #only first 3 to reduce computational effort
        print(f"removed {before - len(tmp)} rows with NVI_STATE_from_ct outside first {first_n_states} US_states")

    if 1: #add quantitative features
        #"est_price" is calculated from "LST_PRCE_AMT", but set to 50000 if "LST_PRCE_AMT" is <=0 or not available
        tmp["est_price"] = tmp["LST_PRCE_AMT"].apply(lambda x: 50000 if x <= 0 or pd.isna(x) else int(round(x)))
        #calculate price for each group
        tmp["model_price"] = tmp.groupby("MODEL_DESC")["est_price"].transform("mean").round().astype(int)
        tmp["segment_price"] = tmp.groupby("SEGMENT_DESC")["est_price"].transform("mean").round().astype(int)
        tmp["brand_price"] = tmp.groupby("MAKE_DESC")["est_price"].transform("mean").round().astype(int)
        tmp = tmp.drop(columns=["est_price"])

    if 1: #print top 5 values of tmp
        print_top_values(tmp, top_n=5, sort=None, add_dtype=True)

    if 1: #remove rows with SLS_COMMERCIAL_FLAG == True or SLS_DEALER_FLAG == True
        print(cross_with_totals(tmp, "SLS_COMMERCIAL_FLAG","SLS_DEALER_FLAG"))
        before=len(tmp)
        tmp = tmp[(tmp["SLS_COMMERCIAL_FLAG"] != "Y") & (tmp["SLS_DEALER_FLAG"] != "Y")] #note: they may be Null
        print(f"removed {before - len(tmp)} rows with SLS_COMMERCIAL_FLAG == True or SLS_DEALER_FLAG == True")

print_brand_reporting(tmp)

if 1: #remove unused, newly generated columns
    rem_col=["is_registered", "is_reported", "remaining_day_of_month"]
    tmp = tmp.drop(columns=rem_col)
    print(f"removed unused columns {rem_col}")

if 1: #save data
    tmp_filename = "tmp_20260205.csv"
    tmp.to_csv(tmp_filename, index=False)
    print(f"saved {tmp_filename} with {len(tmp)} rows")

if 1: #plot data
    plot_multiple_timelines(tmp,
                            date_cols=None,
                            start_date_str="24-10-01",
                            invalid_date_threshold="1900-01-01",
                            replacement_date="2024-09-30",
                            filename_prefix="tmp",
                            drop_last=False)


if 0: #time difference SLS_OWNSHP_DT vs NVI_OWNSHP_DT
    diff = (pd.to_datetime(tmp['SLS_OWNSHP_DT']) - pd.to_datetime(tmp['NVI_OWNSHP_DT'])).dt.round('D').dt.days
    diff_counts = diff.value_counts().sort_index()
    diff_counts.name = 'count'
    diff_counts.index.name = 'days_diff'
    print("Time difference SLS_OWNSHP_DT - NVI_OWNSHP_DT (days):")
    print(diff_counts.to_string())

print("done")
# breakpoint()
print()

