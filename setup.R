# Install the tested R packages into this project's .Rlib directory.
# Run from the replication repository root: Rscript setup.R
lib <- file.path(getwd(), '.Rlib')
dir.create(lib, showWarnings=FALSE)
.libPaths(c(lib,.libPaths()))
options(repos=c(CRAN='https://cloud.r-project.org'))
if (!requireNamespace('remotes',quietly=TRUE)) install.packages('remotes',lib=lib)
for (pkg in c('plm','urca')) {
  version <- if (pkg=='plm') '2.6-7' else '1.3-4'
  want <- if (pkg=='plm') '2.6.7' else '1.3.4'
  if (!requireNamespace(pkg,quietly=TRUE) || as.character(packageVersion(pkg)) != want)
    remotes::install_version(pkg,version=version,lib=lib,upgrade='never',dependencies=NA)
}
cat('Installed. Run with R_LIBS_USER pointing to',lib,'\n')
