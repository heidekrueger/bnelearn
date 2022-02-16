library(tidyverse)
library(latex2exp)

#### utility function with discontinuity ####

xd = 5

left <- function(x){
  if_else(x<=xd, 0, NULL)
}

right <- function(x){
  if_else(x>=xd, 10-x, NULL)
}

full <- function(x){
  10-x
}

df <- tibble(x=seq(0,12,.0001))

g <- ggplot(df,aes(x)) + 
  stat_function(fun=full, col='lightgrey', linetype='dashed') +
  stat_function(fun=left) + 
  stat_function(fun=right) + 
  #geom_vline(xintercept = xd, col='lightgrey', linetype='dashed') +
  scale_y_continuous(breaks = c(0,10), minor_breaks = NULL, limits = c(0,10), labels=c('0', TeX('v_i'))) +
  scale_x_continuous(breaks=c(0, xd, 10), minor_breaks = NULL, limits = c(0,10.1), labels = c('0', TeX('$b^{(1)}_{-i}$'), TeX('v_i'))) +
  theme_minimal() +
  labs(x='bid of player i', y = 'utility of player i') +
  theme(axis.text = element_text(size=14, face='bold'), axis.title = element_text(size=14, face='bold'))

ggsave('R/figures/utility_discontinuity.eps', device = 'eps', width=8, height = 6, units='in', dpi=300)
