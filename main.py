#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:24:25 2018

@author: xiaowei
"""
import pandas as pd
from mongo_connection import get_mongo_data
from mysql_connection import get_sql_data
import constants as c
import merge_compute as mc
import clustering as cl
import write_out as wo
#%% pull data

# ssh -L 9998:10.128.1.43:27017 xwei@10.128.1.27
get_mongo_data()

# ssh -L 9999:dbreplica.us-east-1.int.smartpaylease.com:3306 xwei@10.128.1.27
get_sql_data(days_b=c.sql_days)

get_sql_data(path=c.raw_location_path, script=c.location_sql)
#%%

#mongo_df = pd.read_pickle(c.raw_mongo_path)
#sql_df = pd.read_pickle(c.raw_sql_path)
#location_df = pd.read_pickle(c.raw_location_path)
#%% preprocessing

merged_df = mc.merge_together()
id_df, feat_df = mc.split_df(merged_df, c.loc_id, c.id_cols, c.both_metrics)
id_df2 = mc.feature_cleaning_id(id_df)
id_df2.to_pickle(c.id_path)
feat_df2 = (feat_df
            .pipe(mc.feature_cleaning_cluster)
            .pipe(mc.feature_engineering_cluster)
            )
feat_df2.to_pickle(c.cluster_data_path)
#%% run clustering

df = pd.read_pickle(c.cluster_data_path)
pipe, X_trans = cl.cluster_preprocess_pca(df)
clusters = cl.search(X_trans, c.clust_nums)
use_clusters = cl.filter_clusters(clusters)
#%% Create top X stores for each cluster

rank_lst = [cl.closest_to_centroid(clust['mod'], X_trans, df.index) for clust in use_clusters]
rank_df = pd.concat(rank_lst)
rank_df.to_pickle(c.rank_data_path)
#%% pca dataframe

pca_df = cl.make_pca_df(use_clusters[-1]['mod'], X_trans, df.index)
pca_df.to_pickle(c.pca_data_path)
pca_inf_df = cl.top_pca_influences(pipe.named_steps['pca'], df)
pca_inf_df.to_pickle(c.pca_inf_data_path)
#%% get all dataframes and output to excel file

feat_df = pd.read_pickle(c.cluster_data_path)
id_df = pd.read_pickle(c.id_path).reset_index().set_index('location_id')
pca_df = pd.read_pickle(c.pca_data_path)
rank_df = pd.read_pickle(c.rank_data_path).sort_values(['num_clusters', 'cluster', 'rnk'])
pca_inf_df = pd.read_pickle(c.pca_inf_data_path)

all_df = feat_df.join(pca_df).join(id_df, how='left')
#rank_joined_df = rank_df.join(id_df, how='left').sort_values(['num_clusters', 'cluster', 'rnk'])
output_dfs = [all_df, rank_df, pca_inf_df]
output_names = ['stores', 'ranks', 'inference']

wo.pdf_to_excel(c.output_folder, output_names, output_dfs)

#wo.pdf_to_csv(c.output_folder, 'ranks', rank_df)
