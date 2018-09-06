import pandas as pd
import numpy as np
from time import time
import warnings
from sklearn.model_selection import train_test_split


def quick_test_model(model, X_set, y):
    # NOTE: I'm using accuracy as the sole metric here because it's the same as either micro or macro f1-score
    # for cases where every example belongs to one and only one category

    warnings.filterwarnings(action='ignore', category=DeprecationWarning) # arises with current lightGBM + numpy

    start = time()
    X_train, X_test, y_train, y_test = train_test_split(X_set, y, test_size=1/3)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_accuracy = round(sum(y_pred_train==y_train)/len(y_train), 3)
    test_accuracy = round(sum(y_pred_test==y_test)/len(y_test), 3)

    print("  train accuracy:", train_accuracy)
    print("  test accuracy:", test_accuracy)
    print("elapsed time:", round((time()-start)/60, 1), 'minutes')

def load_and_clean(gtd_raw_df):
    gtd_df = gtd_raw_df.copy()

    # drop the 6 rows with 'specificity' nulls - these should be impossible, since 'unknown' is included
    gtd_df.dropna(subset=['specificity'], inplace=True)

    # 'compclaim' is corrupted: despite having 'no' and 'unknown' options, it's overwhelmingly null with no explanation
    gtd_df.drop('compclaim', axis=1, inplace=True)

    # for some columns, -9 purely indicates 'unknown' and is precisely the same as Nan
    gtd_df['claimed'] = gtd_df.claimed.replace(-9, np.nan)
    gtd_df['ransom'] = gtd_df.ransom.replace(-9, np.nan)
    gtd_df['ishostkid'] = gtd_df.ishostkid.replace(-9, np.nan)
    gtd_df['doubtterr'] = gtd_df.doubtterr.replace(-9, np.nan)
    gtd_df['nperps'] = gtd_df.nperps.replace(-99, np.nan)
    gtd_df['nperpcap'] = gtd_df.nperpcap.replace(-99, np.nan)

    # 20 unknown months are entered as 0s
    gtd_df['imonth'] = gtd_df.imonth.replace(0, np.nan)

    # drop numerical category codes for categorical string variables (redundant)
    numerical_codes = [
        'natlty1',
        'region',
        'country',
        'attacktype1',
        'attacktype2',
        'attacktype3',
        'targtype1',
        'targtype2',
        'targtype3',
        'targsubtype1',
        'targsubtype2',
        'targsubtype3',
        'weaptype1',
        'weaptype2',
        'weaptype3',
        'weaptype4',
        'weapsubtype1',
        'weapsubtype2',
        'weapsubtype3',
        'weapsubtype4',
        'alternative',
        'hostkidoutcome',
        'propextent',
        'claimmode'
    ]
    gtd_df.drop(numerical_codes, axis=1, inplace=True)

    # add total_null_ct column (done after the above so num-coded categorical features aren't counted twice)
    gtd_df['total_null_ct'] = gtd_df.isnull().sum(axis=1)

    # since there are no iyear nulls, and only 20 imonth nulls, and day of month won't likely be useful,
    # we can drop approxdate altogether:
    gtd_df.drop('approxdate', axis=1, inplace=True)

    # fill nulls in alternative_txt (coded alt explanation, e.g. State Actors, for doubted cases)
    # with new category 'None'
    gtd_df['alternative_txt'] = gtd_df['alternative_txt'].fillna('None')

    # for instances with no longitude or latitude, read in the central longitude and latitude
    # of the country of occurrence (from an external datasheet), but flag as approximate
    gtd_df['coordinates_are_approx'] = gtd_df.longitude.isnull()
    coordinates_df = pd.read_csv('country_coordinates2.csv')[['name','latitude','longitude']]
    coordinates_df.rename(columns={'name':'country_txt','latitude':'latitude2','longitude':'longitude2'}, inplace=True)
    gtd_df = gtd_df.merge(coordinates_df, on='country_txt', how='left')
    gtd_df['longitude'] = gtd_df.longitude.fillna(gtd_df.longitude2)
    gtd_df['latitude'] = gtd_df.latitude.fillna(gtd_df.latitude2)
    gtd_df.drop(['latitude2', 'longitude2'], axis=1, inplace=True)
    # for reasons unclear, the one 'St. Kitts and Nevis' null won't fill even when the name is directly copied and pasted
    # into this spreadsheet.  We'll manually fill it here:
    gtd_df.at[59662,'latitude'] = 17.357822
    gtd_df.at[59662,'longitude'] = -62.783998
    gtd_df.isnull().sum().sum()

    # boolean-ize and drop columns that are overwhelmingly null and have many categories:
    gtd_df['has_second_attacktype'] = ~gtd_df.attacktype2_txt.isnull()
    gtd_df['is_related'] = ~gtd_df.related.isnull()
    gtd_df.drop(['related','attacktype2_txt', 'attacktype3_txt'], axis=1, inplace=True)

    # combine nhours and ndays of hostage situation
    gtd_df['nhours'] = gtd_df.nhours + gtd_df.ndays*24
    gtd_df.drop('ndays', axis=1, inplace=True)

    # convert hostages released to hostages unreleased so that we can fill with 0:
    gtd_df['n_unreleased'] = gtd_df['nhostkid'] - gtd_df['nreleased']
    gtd_df.drop('nreleased', axis=1, inplace=True)

    # fill 0 for hostage-related numbers where no hostages taken
    hostage_num_cols = [
        'nhours',
        'nhostkid',
        'nhostkidus',
        'ransomamt',
        'ransomamtus',
        'ransompaid',
        'ransompaidus',
        'n_unreleased'
    ]
    gtd_df[hostage_num_cols] = gtd_df[hostage_num_cols].fillna(0)

    # drop sparse & unhelpful secondary / tertiary claim info - which is almost entirely null:
    empty_cols = [
        'claim2',
        'claimmode2',
        'claim3',
        'claimmode3',
        'natlty2',
        'natlty3',
        'guncertain2',
        'guncertain3'
    ]
    gtd_df.drop(empty_cols, axis=1, inplace=True)

    # fill 'none' for categorical extent of property damage (includes 'unknown' already; blanks are for no damage)
    gtd_df['propextent_txt'] = gtd_df['propextent_txt'].fillna('none')

    # for now we'll also drop string 'resolution' (date of resolution of extended incident) because it's a unique-valued
    # string; we could later convert it to a number-of-days interval
    gtd_df.drop('resolution', axis=1, inplace=True)

    # drop all other string columns with null counts > 100K, with a few exceptions
    significant_though_null_cols = ['motive', 'location', 'propcomment', 'resolution']
    for col in gtd_df.select_dtypes('object').columns:
        if gtd_df[col].isnull().sum() > 100000 and col not in significant_though_null_cols:
            gtd_df.drop(col, axis=1, inplace=True)

    # fix a single wild longitude error
    if gtd_df.loc[17658, 'longitude'] != -86185896.0:
        print("WARNING: index has changed.  Transform longitude for eventid 198212240004")
    # accessed by index because pandas won't let you change data by boolean accessing
    gtd_df.loc[17658, 'longitude'] = gtd_df.loc[17658, 'longitude'] / -10e5

    gtd_prepped_df = gtd_df.copy()

    # *** DEALING WITH TEXT COLUMNS ***

    # drop the following columns because they're long strings (unique, not categories)
    # we'll save them in a separate df for later feature engineering
    text_cols = [
        'summary', #  textual summary of event
        'motive', # reported motive of attack
        'weapdetail',  # additional notes on weapon
        'location', # additional notes on location
        'target1', # name of target
        'scite1', # source citation 1
        'propcomment'
    ]
    text_df = gtd_prepped_df[text_cols]
    gtd_prepped_df.drop(text_cols, axis=1, inplace=True)

    # *** DEALING WITH HYPERSPECIFIC CATEGORICAL COLUMNS ***

    # list categorical columns with too many distinct values for dummying (ranging 100-3000 vals)
    # NOTE: 'country_txt' at 205 distinct values, is excepted because of its obvious importance
    hyperspecific_cols = [
        'natlty1_txt',      # 210 distinct values, but almost entirely identical to 'country_txt'
        'provstate',        # useful, but 2855 distinct values
        'city',             # useful, but 36672 distinct values
        'targsubtype1_txt', # 112 distinct values, with dwindling tail of value counts
    ]
    # fill their nulls with 'unknown'
    for col in hyperspecific_cols:
        gtd_prepped_df[col] = gtd_prepped_df[col].fillna('Unknown')

    # convert hyperspecific features to the X most common values (X determined by manual dropoff-analysis) or 'other'
    def top_n_cats(series, n=10, keep_nulls=False):
        top_n = series.value_counts().index[:n]
        if keep_nulls:
            top_n = list(top_n) + [np.nan]
        return pd.Series([val if val in top_n else 'other' for val in series])

    hyperspecific_tups = [
        ('provstate', 15),
        ('city', 15),
        ('targsubtype1_txt', 20)
    ]

    for tup in hyperspecific_tups:
        gtd_prepped_df[tup[0]+'_common'] = top_n_cats(gtd_prepped_df[tup[0]], n=tup[1])

    # we'll save the original columns for later feature engineering etc. before dropping them
    hyperspecific_df = gtd_prepped_df[hyperspecific_cols + ['gname']].copy()
    gtd_prepped_df.drop(hyperspecific_cols, axis=1, inplace=True)

    # *** OTHER MISCELLANEOUS ***

    # fill 'weapsubtype1_txt' with 'Unknown' (only 30 categories, so no need to commonize)
    gtd_prepped_df['weapsubtype1_txt'] = gtd_prepped_df['weapsubtype1_txt'].fillna('Unknown')

    # 'corp1' for name of target org. is like the above columns, but it's alread encapsulated by 'targsubtype1'
    # so we'll just drop it
    hyperspecific_df['corp1'] = gtd_prepped_df['corp1']
    gtd_prepped_df.drop('corp1', axis=1, inplace=True)

    gtd_filled_df = gtd_prepped_df.copy()

    gtd_filled_df['imonth'] = gtd_filled_df['nkill'].fillna(0)

    # fill 'guncertain' with 1 (means attributed group isn't a certain attribution)
    gtd_filled_df['guncertain1'] = gtd_filled_df['guncertain1'].fillna(1)

    # median-impute casualties - they're only left blank if there is evidence of some,
    # (so it's non-zero), but there's insufficient evidence to give a number
    casualty_cols = [
        'nkill',
        'nkillus',
        'nkillter',
        'nwound',
        'nwoundus',
        'nwoundte'
    ]
    for col in casualty_cols:
        gtd_filled_df[col] = gtd_filled_df[col].fillna(gtd_filled_df[gtd_filled_df[col] > 0][col].median())

    # fill flags with 0 where unmarked probably means no:
    flag_vars = ['multiple', 'claimed', 'ransom', 'doubtterr', 'ishostkid']
    gtd_filled_df[flag_vars] = gtd_filled_df[flag_vars].fillna(0)

    # fill propvalue with median prop damage for cases where there was prop damage
    gtd_filled_df['propvalue'] = gtd_filled_df['propvalue'].fillna(gtd_filled_df[gtd_filled_df.propvalue > 0].propvalue.median())
    gtd_filled_df['propvalue'] = gtd_filled_df['propvalue'] * gtd_filled_df['property'].replace(-9, 0)

    # num perpetrators would be useful, but it's overwhelmingly null and there's no good way to impute
    gtd_filled_df.drop(['nperps', 'nperpcap'], axis=1, inplace=True)

    # confirm that we have no nulls whatsoever remaining:
    gtd_filled_df.isnull().sum().sum()

    non_domestic_target = [gtd_df.natlty1_txt[i] if gtd_df.natlty1_txt[i] != gtd_df.country_txt[i] else np.nan \
        for i in range(len(gtd_df.natlty1_txt))]

    gtd_df['non_domestic_target'] = top_n_cats(pd.Series(non_domestic_target), n=30, keep_nulls=True)
    gtd_prepped_df['non_domestic_target'] = top_n_cats(pd.Series(non_domestic_target), n=30, keep_nulls=True)
    gtd_filled_df['non_domestic_target'] = top_n_cats(pd.Series(non_domestic_target), n=30, keep_nulls=False)

    return gtd_df, gtd_prepped_df, gtd_filled_df

def compile_profiles(dummy_df, scaler=StandardScaler(), importance_quotient=2, include_singletons=True):
    '''
    Note: it's important to scale BEFORE performing groupby because the groupby runs multiple agg functions
    on some columns that should have a fixed relation to each other (e.g., median, max, var)
    Also: importance_quotient upscales important columns after scaling to increase their clustering influence
    '''
    start = time()

    if not include_singletons:
        dummy_df = dummy_df[dummy_df.gname in dummy_df.gname.value_counts()[1:1826]]

    # dummy out the quasi-binary columns that also have a lot of -9 flags, so that they're basically nan-as-cat
    dummy_temp_df = dummy_df.drop(['gname', 'eventid', 'iday'], axis=1)
    for col in ['INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', 'vicinity', 'property']:
        dummy_temp_df[col] = dummy_temp_df[col].astype(str)
    dummy_temp_df = pd.get_dummies(dummy_temp_df)
    dummy_temp_df['gname'] = dummy_df.gname

    # set agg functions
    agg_funcs = {}
    for col in dummy_temp_df.select_dtypes(['uint8', 'bool']).columns:
        agg_funcs[col] = ['mean']
    for col in dummy_temp_df.select_dtypes(['int64', 'float64']).columns:
        agg_funcs[col] = ['median', 'max', 'var']
    dummy_temp_df['attacks'] = dummy_temp_df.gname
    agg_funcs['attacks'] = 'count'

    # perform groupy and flatten multi_index
    group_profiles_df = dummy_temp_df.groupby('gname', as_index=False).agg(agg_funcs)
    group_profiles_df.columns = ['_'.join(col) for col in group_profiles_df.columns]
    # a strange pandas bug sometimes adds a _ to the end of gname, so:
    group_profiles_df.columns = [col if not col.endswith('_') else col[:-1] for col in group_profiles_df.columns]

    # median-impute variance for singletons (otherwise is NaN)
    if include_singletons:
        for variance_col in [col for col in group_profiles_df.columns if col.endswith('_var')]:
            group_profiles_df[variance_col] = group_profiles_df[variance_col].fillna(
                group_profiles_df[variance_col].median())

    # apply scaler
    columnlist = list(group_profiles_df.columns)
    columnlist.remove('gname')
    group_profiles_df = pd.concat(
        [group_profiles_df.gname,
         pd.DataFrame(scaler.fit_transform(group_profiles_df.drop('gname', axis=1)), columns=columnlist)],
        axis=1)

    for dummy_col in dummy_temp_df.select_dtypes(['uint8', 'bool']).columns:
        group_profiles_df[dummy_col + '_mean'] = group_profiles_df[dummy_col + '_mean'] * 3
        group_profiles_df['attacks_count'] = group_profiles_df.attacks_count * 3
    # then give a little extra weight to some features we know are particularly important:
    for orig_col in ['country_txt', 'INT_LOG', 'latitude', 'longitude', 'attacks_count']:
        for col in [col for col in group_profiles_df.columns if orig_col in col]:
            group_profiles_df[col] = 2 * group_profiles_df[col]

    del dummy_temp_df
    return group_profiles_df
