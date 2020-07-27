###################### Analyse detailed data#################
## Compute stopping criterium and times for any bidder type
# Compute \hat{L} and in which epochs the stopping criteria are met for each subrun
tb_stop <- tb_full %>% 
  pivot_wider(names_from = tag, values_from = value) %>% 
  filter(subrun %in% type_names,
         epoch%%stop_criterium_interval==0,
         epoch <= results_epoch) %>% 
  mutate(# Compute \hat{L} = 1 - u(beta_i)/u(BR)
         eval_util_loss_rel_estimate = 1 - eval_utilities/(eval_utilities+eval_util_loss_ex_ante)) %>% 
  group_by(run,subrun) %>% 
  mutate(stopping_crit_diff = (pmax(eval_util_loss_ex_ante, 
                                if_else(is.na(lag(eval_util_loss_ex_ante, n=1L)), 9, 
                                        lag(eval_util_loss_ex_ante, n=1L)),
                                if_else(is.na(lag(eval_util_loss_ex_ante, n=2L)), 99, 
                                        lag(eval_util_loss_ex_ante, n=2L)),na.rm=TRUE) - 
                                 pmin(eval_util_loss_ex_ante,
                                       if_else(is.na(lag(eval_util_loss_ex_ante, n=1L)), 9, 
                                               lag(eval_util_loss_ex_ante, n=1L)),
                                       if_else(is.na(lag(eval_util_loss_ex_ante, n=2L)), 99, 
                                               lag(eval_util_loss_ex_ante, n=2L)), na.rm = TRUE))) %>% 
  ungroup() %>% 
  mutate(stop_diff_1 = if_else(stopping_crit_diff < stop_criterium_1,TRUE, FALSE),
         stop_diff_2 = if_else(stopping_crit_diff < stop_criterium_2,TRUE, FALSE)) %>% 
  select(c(names(.)[1:3],"stopping_crit_diff","stop_diff_1","stop_diff_2"))

# compute times and epochs until convergence and total runtime
tb_runtime <- tb_full %>% 
  pivot_wider(names_from = tag, values_from = value) %>% 
  select(c(names(.)[1:4],"eval_overhead_hours")) %>% 
  filter(subrun %in% c(".")) %>% 
  drop_na(.) %>% 
  group_by(run, subrun) %>% 
  mutate(runtime = wall_time - wall_time[1] - eval_overhead_hours) %>% 
  ungroup() %>% 
  select(c("run","epoch","runtime"))

tb_stop <- merge(tb_runtime, tb_stop, by=(c("run", "epoch"))) 

# Filter for epochs in which all participants match criterium
tb_stop <- tb_stop %>% 
  arrange(epoch) %>% 
  group_by(run,epoch) %>% 
  mutate(stop_diff_1 = all(stop_diff_1),
         stop_diff_2 = all(stop_diff_2)) %>% 
  arrange(run) %>% 
  ungroup() %>% 
  # Filter the last row and the ones where stop_crit is fulfilled
  filter(epoch == results_epoch |
         stop_diff_2==TRUE |
         stop_diff_1==TRUE) %>% 
  group_by(run) %>% 
  # Filter when stop_crit was first fulfilled
  filter(epoch %in% c(results_epoch,epoch[min(which(stop_diff_2 == TRUE))],
                    epoch[min(which(stop_diff_1 == TRUE))])) %>% 
  ungroup()

## Bring into format
# get only first occasions of stopping criteria in wide format
tb_stop_print <- tb_stop %>% 
  filter(subrun == type_names[1]) %>% 
  group_by(run) %>% 
  mutate(subrun = ".",
         stop_diff_1_e = stop_diff_1 * epoch,
         stop_diff_2_e = stop_diff_2 * epoch,
         stop_diff_1_e = if_else(stop_diff_1_e>min(stop_diff_1_e[stop_diff_1_e>0]),
                                 0,stop_diff_1_e),
         stop_diff_2_e = if_else(stop_diff_2_e>min(stop_diff_2_e[stop_diff_2_e>0]),
                                 0,stop_diff_2_e),
         stop_diff_1_t = (stop_diff_1 * runtime)/60,
         stop_diff_1_t = if_else(stop_diff_1_t>min(stop_diff_1_t[stop_diff_1_t>0]),
                                 0,stop_diff_1_t),
         stop_diff_2_t = (stop_diff_2 * runtime)/60,
         stop_diff_2_t = if_else(stop_diff_2_t>min(stop_diff_2_t[stop_diff_2_t>0]),
                                 0,stop_diff_2_t),
         time = if_else(runtime<max(runtime),
                        0,max(runtime)/60)) %>%
  ungroup() %>% 
  select(-c("runtime","stopping_crit_diff","stop_diff_2","stop_diff_1","epoch"))
# transform to long format
tb_stop_print <- tb_stop_print %>% 
  pivot_longer(cols=-c(run,subrun),names_to="tag",values_to="value",values_drop_na = TRUE) %>% 
  filter(value>0) %>% 
  select(-c("run")) %>% 
  group_by(subrun,tag) %>% 
  summarize(avg = mean(value),
            std = sqrt(var(value)),
            cnt = n())

### Analyse eval data###############
# Select only end and widen
tb_eval <- tb_full %>% 
  filter(epoch == results_epoch) %>% 
  pivot_wider(names_from = tag, values_from = value)

# # TMP!!: For Journal eval
# tb_eval %>%
#  mutate(utility_bne = eval_utility_vs_bne + eval_epsilon_absolute) %>%
#  pivot_longer(colnames(.)[4:(length(colnames(.)))], names_to="tag",
#               values_to="value", values_drop_na = TRUE) %>%
#  group_by(subrun, tag) %>%
#  summarise(avg = mean(value),
#            std = sd(value)) %>%
#  filter(tag %in% c("eval_utilities","eval_utility_vs_bne", "utility_bne", 
#                    "eval_epsilon_relative", "eval_L_2")) %>%
#  mutate(avg = sprintf("%0.4f", avg),
#         std = sprintf("%0.4f", std)) %>%
#  print(n=28)

# If we calculated a more exact bne utility, use that
if(exists("utility_in_bne_exact")){
  tb_eval <- tb_eval %>% 
    mutate(utility_bne = if_else(subrun == type_names[1], utility_in_bne_exact[1],
                                 if_else(subrun == type_names[2], utility_in_bne_exact[2], 9999)),
           utility_bne = replace(utility_bne, utility_bne==9999, NA),
           # Compute exact epsilone 
           eval_epsilon_absolute = utility_bne - eval_utility_vs_bne,
           eval_epsilon_relative = 1 - (eval_utility_vs_bne/utility_bne))
}
# More general if no BNE is known!
if(known_bne == F){
  tb_eval <- tb_eval %>% 
    mutate(eval_util_loss_rel_estimate = 1 - eval_utilities/(eval_utilities+eval_util_loss_ex_ante)) %>% 
    # Select only necessary columns
    select(c("run","subrun","epoch","eval_utilities", "eval_util_loss_ex_ante",
             "eval_util_loss_ex_interim","eval_util_loss_rel_estimate")) 
}else{
  tb_eval <- tb_eval %>% 
    mutate(# temporary estimate for: Compute \hat{L} = 1 - u(beta_i)/u(BR)
           eval_util_loss_rel_estimate = 1 - eval_utilities/(eval_utilities+eval_util_loss_ex_ante)) %>% 
    # Select only necessary columns
    select(c("run","subrun","epoch","eval_utilities", "eval_epsilon_absolute", "eval_epsilon_relative", "eval_L_2", "eval_L_inf",
             "eval_util_loss_ex_ante", "eval_util_loss_ex_interim",
             "eval_util_loss_rel_estimate"))
}
tb_eval <- tb_eval %>% 
  pivot_longer(colnames(.)[4:length(colnames(.))], names_to="tag", 
               values_to="value", values_drop_na = TRUE) %>% 
  group_by(subrun, tag) %>% 
  summarise(avg = mean(value),
            std = sqrt(var(value))) %>% 
  print(n=28)

########Bind tables and format output#########
tb_final = rbind(tb_eval,tb_stop_print)
tb_final

# TODO: create nice latex export table
tb_final %>% 
  # Create to format 10^-4
  mutate(
         #avg = avg * 10^4,
         #std = std * 10^4,
         avg = sprintf("%0.4f", avg)) %>% 
         #std = sprintf("%0.5f", std)) %>% 
  pivot_wider(id_cols=subrun,names_from=tag, values_from=avg) %>% 
  select(subrun, contains("eval_epsilon_relative"), contains("eval_L_2"), eval_util_loss_ex_ante, 
         eval_util_loss_ex_interim, eval_util_loss_rel_estimate, contains("stop_diff_2_e"), contains("stop_diff_1_e"), time) %>%
  arrange(-row_number()) %>% 
  kable("latex", booktabs = T)

# # ###########For Split Award skip line 130 ff and run this#########
# if(experiment == "SplitAward"){
#   tb_eval <- tb_eval %>% 
#     mutate(# temporary estimate for: Compute \hat{L} = 1 - u(beta_i)/u(BR)
#       eval_util_loss_rel_estimate = 1 - eval_utilities/(eval_utilities+eval_util_loss_ex_ante)) %>% 
#     # Select only necessary columns
#     select(c("run","subrun","epoch","eval_utilities", 
#              "eval_epsilon_absolute_bne1", "eval_epsilon_relative_bne1", "eval_L_2_bne1", "eval_L_inf_bne1",
#              "eval_epsilon_absolute_bne2", "eval_epsilon_relative_bne2", "eval_L_2_bne2", "eval_L_inf_bne2", 
#              "eval_util_loss_ex_ante", "eval_util_loss_ex_interim", "eval_util_loss_rel_estimate")) %>% 
#     pivot_longer(colnames(.)[4:length(colnames(.))], names_to="tag", 
#                  values_to="value", values_drop_na = TRUE) %>% 
#     group_by(subrun, tag) %>% 
#     summarise(avg = mean(value),
#               std = sqrt(var(value))) %>% 
#     print(n=28)
# }



