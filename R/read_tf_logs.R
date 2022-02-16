library(tidyverse)

# furr parallelization fails because of some stupid Rstudio incompability
#but this should work when running this as a script
#library(furrr) # parallel purrr
#plan(multicore, workers =32)
#supportsMulticore()

# RProtobuf is an R library for google protocoll buffers -- doesn't work with reticulate
# see https://github.com/eddelbuettel/rprotobuf
#library(RProtoBuf)

# devtools::install_github('rstudio/tensorflow')
# load r tensorflow package and point it to existing tensorflow installation in
# conda envrionment named r-tensorflow
library(tensorflow)

use_condaenv('r-tensorflow', conda = '/opt/anaconda/anaconda3/bin/conda', required = T)
tf_config()
tb = import('tensorboard')

# Functions for reading tf-logs-----------------------------------------------------------------


#' Convert a python `ScalarEvent` to a named vector in R.#'
#' @param x A python ScalarEvent provided by a #'
#' @return x converted to named r-vector
scalar_event_to_r <- function(x){
  x$`_asdict`()
}

#' Return a dataframe of all scalar events in a tag#'
#' @param tag A tag (character)
#' @param acc A tensorboard EventAccumulator#'
#' @return Dataframe of all events matching the tag in the accumulator
tag_to_df <- function(tag, acc){
  #print(tag)
  events <- acc$Scalars(tag)
  events %>% 
    map_dfr(scalar_event_to_r) %>% 
    mutate(tag=tag) %>% 
    separate(tag, c('supertag', 'tag'), sep='/', fill='left',extra='merge') %>% 
    mutate_at(vars(wall_time), lubridate::as_datetime)
}

#' Returns all scalar events in a tensorflow event file in a dataframe
#' @param file_path 
#' @param run_name Name of the run associated with the file
#' @return a dataframe
file_to_df <- function(file_path, run_name){
  
  accumulator = tb$backend$event_processing$event_accumulator$EventAccumulator(file_path)
  accumulator$Reload()
  tags <-accumulator$Tags()$scalars
  
  closure = function(tag){
    tag_to_df(tag, accumulator)
  }
  
  tags %>% map_dfr(.,closure) %>% mutate(run=run_name)
}

#' Gets a data frame of all events in a directory.
#' 
#' @param path Path to the directory. Should only have 1 level of subdirectories corresponding to
#'             runs. All event files in each subdirectory will be 
#' @return Tibble of all logged scalars in the experiment folder.
experiment_dir_to_df <- function(path){
  assertthat::are_equal(str_sub(path, start=-1), '/') #path needs to end in /, otherwise problems below
  files = list.files(path, pattern='.*events\\.out\\.tfevents.*', recursive=T)
  run_names =files %>% str_split_fixed('/', 2) %>% .[,1]
  
  filepaths = paste0(path,files)
  map2_dfr(filepaths, run_names, file_to_df)
}
games = c('RPS','JG')#c('PD','MP','BoS','RPS','JG')

for (g in games){
  path = paste('bnelearn/experiments/notebooks/matrix/',g,'/',sep="") #test/CopyOfFP_2019-08-08 Thu 14:38:51/'
  
  {
    library(tictoc)
    tic()
    write_csv(experiment_dir_to_df(path),paste('bnelearn/experiments/data/',g,'.csv',sep=""))
    toc()
  }
  
}

