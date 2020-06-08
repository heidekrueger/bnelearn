# Title     : TODO
# Objective : TODO
# Created by: 
# Created on: 5/10/20


library(reticulate)
use_condaenv("bnelearn", conda = "auto")
pd <- import("pandas")

aggregate_f_name <- '/home//bnelearn/experiments/single_item/second_price/normal/symmetric/risk_neutral/2p/2020-05-10 Sun 17.33/aggregate_log.csv'
full_f_name <- '/home//bnelearn/experiments/single_item/second_price/normal/symmetric/risk_neutral/2p/2020-05-10 Sun 17.33/full_results.pkl'
aggreagte_df <- pd$read_csv(aggregate_f_name)
full_df <- pd$read_pickle(full_f_name)

print(aggreagte_df)
