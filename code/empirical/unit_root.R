args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 1)
suppressPackageStartupMessages(library(plm))
out <- args[1]
rows <- lapply(c('EP','DP','SP'), function(name) {
    x <- as.matrix(read.csv(file.path(out,paste0('unit_root_input_',name,'.csv'))))
    test <- purtest(x, test='Pm', exo='none', lags='SIC', pmax=10,
                    Hcons=TRUE, fixedT=TRUE)
    data.frame(variable=name,statistic=unname(test$statistic$statistic),
               pvalue=unname(test$statistic$p.value))
})
write.csv(do.call(rbind,rows),file.path(out,'unit_root.csv'),row.names=FALSE)
