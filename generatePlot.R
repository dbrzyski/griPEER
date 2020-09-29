rm(list=ls())

## Libraries
library(ggplot2)
library(dplyr)

# Make sure functions from dplyr are acting as functions from dplyr 
filter    <- dplyr::filter
mutate    <- dplyr::mutate
group_by  <- dplyr::group_by
summarize <- dplyr::summarize
select    <- dplyr::select

## Define project directory path 
results.path  <- "path to data directory"

## Setting
gg.base_size <- 9



setwd(results.path )

b_est            <- read.table('b_est_vec.txt') 
asymp_conf_lower <- read.table('asymp_conf_lower.txt')
asymp_conf_upper <- read.table('asymp_conf_upper.txt')
boot_conf_lower  <- read.table('boot_conf_lower.txt')  
boot_conf_upper  <- read.table('boot_conf_upper.txt') 
b_idx            <- read.table('b_idx_vec.txt') 
sc_matrix_name   <- c(rep("Empty", times = 68), rep("Masked FA", times = 68), rep("Masked DC", times = 68))
res.df           <- data.frame(b_est, asymp_conf_lower, asymp_conf_upper, boot_conf_lower, boot_conf_upper, b_idx, sc_matrix_name)
colnames(res.df) <- c("b_est", "asymp_conf_lower", "asymp_conf_upper", "boot_conf_lower", "boot_conf_upper", "b_idx", "sc_matrix_name")



# add brain regions signatures 
regions_full_name   <- unlist(read.table(paste0(results.path, "regions_full_name.txt"), 
                                         sep = "\t", header = FALSE, stringsAsFactors = FALSE))
regions_sgn         <- unlist(read.table(paste0(results.path, "regions_sgn.txt"), 
                                         sep = "\t", header = FALSE, stringsAsFactors = FALSE))
regions.df          <- data.frame(regions_full_name = factor(regions_full_name, levels =  regions_full_name),
                                  regions_sgn = factor(regions_sgn, levels =  regions_sgn), 
                                  b_idx = (1:66 + 2))
res.df              <- res.df %>% left_join(regions.df, by = "b_idx")

# Filter out non-b coefficients
res.df.f <- res.df %>% filter(! b_idx%in% c(1,2))

# add statistical significance
res.df.f <- res.df.f %>%  mutate(is_boot_sgnf  = sign(boot_conf_lower * boot_conf_upper) > 0,
                                 is_asymp_sgnf = sign(asymp_conf_lower * asymp_conf_upper) > 0)

res.df.f$sc_matrix_name <- factor(as.character(res.df.f$sc_matrix_name),
                                  levels = sort(unique(res.df.f$sc_matrix_name)),
                                  labels = c("Empty", "Masked FA", "Masked DC"))

scale_manual_VALS_RIDGE  <- c("grey" , "#00BF7D" ,"#00B0F6")

## Generate the plot
res.df.f2 <- res.df.f
res.df.f2$sc_matrix_name <- factor(as.character(res.df.f2$sc_matrix_name),
                                   levels = c("Empty", "Masked FA", "Masked DC"),
                                   labels = c("Empty", "Masked FA", "Masked DC"))
res.df.f2.sgnf.boot  <- res.df.f2 %>% filter(is_boot_sgnf == TRUE)
res.df.f2.sgnf.asymp <- res.df.f2 %>% filter(is_asymp_sgnf == TRUE)

ggplot(res.df.f2, aes(x = regions_full_name, y = b_est, 
                      color = sc_matrix_name, group = 1, fill = sc_matrix_name)) +
  geom_hline(yintercept = 0) +
  geom_ribbon(aes(ymin = boot_conf_lower, ymax = boot_conf_upper), alpha = 0.3, color = "black", size = 0.1) +
  geom_line() +
  geom_vline(data = res.df.f2.sgnf.boot, aes(xintercept = b_idx - 2), color = "red", size = 0.8) +
  facet_grid(sc_matrix_name ~ ., scales = "free") +
  theme_bw(base_size = gg.base_size) +
  theme(text = element_text(size=13), plot.margin = unit(c(0.2,0,0.2,1),"cm"), axis.title=element_text(size=14,face="bold"), axis.text.x = element_text(angle = 55, hjust = 1), legend.position="top", legend.title = element_text(size = 14)) +
  labs(x = "Brain regions", y = "griPEER estimate", color = "Connectivity Matrix:   ", group ="Connectivity Matrix:   ", fill ="Connectivity Matrix:   ") +
  scale_fill_manual(values=scale_manual_VALS_RIDGE, labels=c("Empty (logistic Ridge) ", "Masked FA  ", "Masked density")) + 
  scale_color_manual(values=scale_manual_VALS_RIDGE, labels=c("Empty (logistic Ridge) ", "Masked FA  ", "Masked density"))

## Save plot

setwd(results.path)
ggsave("HIV_males_results.pdf", width=12, height=6.3)



