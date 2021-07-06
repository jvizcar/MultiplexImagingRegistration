library(pacman)
p_load(plyr)
p_load(dplyr)
p_load(ggplot2)
p_load(latex2exp)
p_load(ggbeeswarm)

files <- c(
      # Non-registered
      "BRCA_hackathon_recreation_noreg_results.csv",
      "HCC1143_hackathon_recreation_noreg_results.csv",
      "Tonsil_hackathon_recreation_noreg_results.csv",
      # Non-rigid registration using target round 1 (i.e., with some marker)
      # from hackathon	
      ## "BRCA_hackathon_recreation_nonrigid_results.csv",
      ## "HCC1143_hackathon_recreation_nonrigid_results.csv",
      ## "Tonsil_hackathon_recreation_nonrigid_results.csv",
      # Non-rigid registration using target round 0 (i.e., background /
      # no marker) by JC
      "BRCA_backgroundreg_results.csv",
      "HCC1143_backgroundreg_results.csv",
      "Tonsil_backgroundreg_results.csv",
      # Rigid results using DAPI
      "BRCA_hackathon_recreation_rigid_results.csv",
      "HCC1143_hackathon_recreation_rigid_results.csv",
      "Tonsil_hackathon_recreation_rigid_results.csv")

df <- ldply(files, .fun = function(file) read.table(file, sep=",", header=TRUE, stringsAsFactors = FALSE))


adjust.entry <- function(df, col, old.entry, new.entry) {
  flag <- df[,col] == old.entry
  df[flag,col] <- new.entry
  return(df)
}

#df <- adjust.entry(df, "Dataset", "BRCA (OHSU)", "Breast")
#df <- adjust.entry(df, "Dataset", "HCC1143 (OHSU)", "HC1143")

# NB: JC inadvertenly switched BRCA and HCC1143 -- switch them back here.
# He notes this in an email from June 29, 2021 (to brian.white@jax.org) with subject
# Join image analysis working group meeting today? (In 45 minutes!)
df <- adjust.entry(df, "Dataset", "BRCA (OHSU)", "HC1143")
df <- adjust.entry(df, "Dataset", "HCC1143 (OHSU)", "Breast")

df <- adjust.entry(df, "Dataset", "Tonsil (OHSU)", "Tonsil")
df <- adjust.entry(df, "Registration.Method", "Nonrigid registration - Background", "Background (Non-rigid)")
df <- adjust.entry(df, "Registration.Method", "Rigid Registration with DAPI", "DAPI (Rigid)")

df$Registration.Method <- factor(df$Registration.Method, levels = c("Unregistered", "DAPI (Rigid)", "Background (Non-rigid)"))

lvls <- c("Breast", "Tonsil")
lvls <- c("Tonsil", "Breast", "HC1143")

## df <- subset(df, Dataset %in% c("Breast", "Tonsil"))

df <- subset(df, Dataset %in% lvls)
df$Dataset <- factor(df$Dataset, levels=lvls)



## We always calculate TRE with respect to R1 -- regardless of how which round we use to derive the registration transform.
## The rigid transformations are derived using R1; hence it has already been excluded from these results (an alignment
## from itself to itself should have zero TRE).
## For symmetry, lets also exclude the TRE for R1 for the background / non-rigid TREs. In principle, this will not be
## zero (since the registration transformation was derived from R0), but it still sounds fishy

df <- subset(df, Moving.Round != "R1")

print(table(df$Moving.Round))

g <- ggplot()
g <- g + geom_boxplot(data = df,
                      aes(x = Registration.Method, y = Mean.Error..um.), outlier.shape = NA)
g <- g + geom_beeswarm(data = df,
                      aes(x = Registration.Method, y = Mean.Error..um.))
## g <- g + facet_wrap(~ Dataset)
# g <- g + facet_wrap(~ Dataset, scale = "free_y")
g <- g + facet_wrap(~ Dataset)
g <- g + theme(axis.text.x = element_text(angle = 45, hjust=1), text = element_text(size = 15))
g <- g + xlab("Registration Method")
g <- g + ylab(TeX("Mean distance from target image ($\\mu{}$m)"))
## Put this back if we exclude Breast
if(!("Breast" %in% df$Dataset)) {
##  g <- g + scale_y_continuous(breaks=c(1, 10, 100, 1000, 2000, 3000))
} else {
print("here\n")
	g <- g + scale_y_continuous(breaks=c(1, 10, 100, 1000, 2000, 3000))
}
g <- g + coord_trans(y="log2")
## g <- g + scale_y_continuous(trans = log2_trans(),
##     breaks = trans_breaks("log2", function(x) 2^x),
##     labels = trans_format("log2", math_format(2^.x)))

png("registration.png")
print(g)
d <- dev.off()

pdf("registration.pdf")
print(g)
d <- dev.off()
