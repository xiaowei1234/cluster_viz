# Channel Clusters

## Introduction

This repository is for the user to gain a better understanding of the product and the retailers. It will also give an understanding of what kind of data are available.

A secondary objective is to provide a list of retailers that are "representative" of the general population and can serve as suitable candidates to conduct testing

## Navigating this Repository

### code folder

*mongo_connection.py* - connect to mongodb and get data (not in this repo)

*mysql_connection.py* - connection to mysql and get data (not in this repo)

*constants.py* - constants used as configurations (not in this repo)

*sql_script* - script to get data from lease_summaries table (not in this repo)

*locations.sql* - script to get data from locations table. Reason this is not rolled into sql_script.sql as a join is because mysqldb is very slow with joins (not in this repo)

*merge_compute.py* - merge together data sources and do cleaning and manipulations

*clustering.py* - meat of code used for clustering/PCA/etc.

*write_out.py* - write out to Excel file

*main.py* - script for executing everything

### Outputs folder

*clustering.xlsx* - file to put into visualization tool like Tableau (not in this repo)
