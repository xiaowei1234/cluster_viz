import pandas as pd
import numpy as np
import constants as c


def merge_together(mongo_path=c.raw_mongo_path, sql_path=c.raw_sql_path, location_path=c.raw_location_path):
    """
    """
    mongo_df = pd.read_pickle(mongo_path)
    mongo_df = mongo_df.loc[pd.notnull(mongo_df['request.location_address.location_id']), :]
    mongo_df[c.loc_id] = mongo_df['request.location_address.location_id'].astype(np.int64)
    sql_df = pd.read_pickle(sql_path)
    location_df = pd.read_pickle(location_path)
    df = (mongo_df
        .merge(sql_df, how='left'
            , left_on='request.order.application_number', right_on='application_number')
        .merge(location_df, how='left', left_on='request.location_address.location_id', right_on='id')
        )
    drops = ['request.order.application_number', 'application_number', 'id', 'request.location_address.location_id']
    return df.drop(drops, axis=1)


def split_df(df, id_col, id_df_cols, both):
    """
    """
    df2 = df.set_index(id_col)
    id_df = df2[id_df_cols + both]
    feat_df = df2.drop(id_df_cols, axis=1)
    return id_df, feat_df


def bool_to_float(series):
    return series.astype(np.float64)


has_val = lambda v: 1 if pd.notnull(v) else 0


def feature_cleaning_id(id_df, agg_metrics=c.id_agg_metrics):
    gby = [col for col in id_df if col not in agg_metrics]
    id_df2 = id_df.copy()
    id_df2[gby] = id_df2[gby].fillna('missing')
    id_df2['more_info'] = bool_to_float(id_df2['extra_variables.is_more_info_flow'])
    more_info = id_df2['extra_variables.is_more_info_flow'].apply(lambda v: v if pd.notnull(v) else False)
    df = id_df2.groupby([c.loc_id] + gby).mean().drop('outgoing_payment_amount', axis=1)
    df2 = id_df2.groupby([c.loc_id] + gby).sum()[['outgoing_payment_amount']].rename(
        columns={'outgoing_payment_amount': 'location_revenue'})
    return df.join(df2)


null_zero = lambda v: 0 if pd.isnull(v) else v


def nan_1_0(val):
    if pd.isnull(val):
        return np.nan
    if val > 0:
        return 0
    return 1


def feature_cleaning_cluster(df):
    """
    """
    df2 = df.copy()
    df2['approvals'] = bool_to_float(df2['decision_engine.decision'])
    df2['itin'] = bool_to_float(df2['request.user.is_itin'])
    df2['other_phone'] = (df2.iphone + df2.galaxys).apply(nan_1_0)
    df2['location_age'] = df2['pmml_variables.fe_days_since_location_created'] / 365
    df2['num_items'] = df2.num_items - df2.str_plan
    df2['cart_limit'] = df2['decision_engine.cart_limit'] + df2['decision_engine.bonus_amount'].apply(null_zero)
    drops = ['decision_engine.decision'
            , 'request.user.is_itin', 'pmml_variables.fe_days_since_location_created'
            , 'str_plan'
            , 'decision_engine.cart_limit', 'decision_engine.bonus_amount']
    renames = {'pmml_variables.fe_age_in_years': 'cust_age', 'outgoing_payment_amount': 'purchase_amount'
            , 'decision_engine.score.points': 'decision_score'}
    return df2.drop(drops, axis=1).rename(columns=renames)


def feature_engineering_cluster(df):
    """
    """
    # df['cart_usage'] = df.purchase_amount / df.cart_limit
    df.loc[df.approvals > 0, 'take'] = df.purchase_amount[df.approvals > 0].apply(lambda v: 1 if pd.notnull(v) else 0)
    df['applications'] = 1
    df_locs_mean = df.groupby(c.loc_id).mean().drop(['applications', 'approvals', 'take'], axis=1)
    df_locs_apps = df[['applications', 'approvals', 'take']].groupby(c.loc_id).sum()
    df_locs_apps['approval_rate'] = df_locs_apps.approvals / df_locs_apps.applications 
    df_locs_apps['take_rate'] = df_locs_apps['take'] / df_locs_apps.approvals
    df_locs_apps.drop(['approvals', 'take'], axis=1, inplace=True)
    df_locs = df_locs_mean.join(df_locs_apps)
    return df_locs.loc[(df_locs.applications > 2) & pd.notnull(df_locs.purchase_amount), :]


if __name__ == '__main__':
    df = merge_together()
    id_df, feat_df = split_df(df, c.loc_id, c.id_cols)
    id_df2 = feature_cleaning_id(id_df)
    feat_df2 = feature_cleaning_cluster(feat_df)
