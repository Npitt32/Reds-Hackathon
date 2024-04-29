setwd("C:/Users/legio/OneDrive/Spring 2024/SAL 358/Reds Hackathon")
library(tidyverse)
library(caret)
library(rsample)
library(vip)
library(dbscan)
library(ROCR)
library(car)

savant_pitch_level <- read_csv("savant_pitch_level_updated.csv")

# Adding same_hand and batter_chase and filtering data
new_statcast <- savant_pitch_level %>%
  mutate(same_hand = case_when(
    (stand == "L" & p_throws == "L") | (stand == "R" & p_throws == "R") ~ 1,
    TRUE ~ 0
  )) %>%
  mutate(batter_chase = ifelse(batter_chase == "batter chased", 1, 
                               ifelse(batter_chase == "batter did not chase", 
                                      0, NA))) %>%
  filter(chase == 1 | waste == 1)

# Making dependent variable a factor and removing NAs
new_statcast$batter_chase <- as.factor(new_statcast$batter_chase)
new_statcast <- new_statcast[complete.cases(new_statcast$pfx_x), ]
new_statcast <- new_statcast[complete.cases(new_statcast$pfx_z), ]
new_statcast <- new_statcast[complete.cases(new_statcast$release_speed), ]

fastballs <- c("4-Seam Fastball", "Sinker")

breaking_balls <- c("Curveball", "Slider", "Knuckle Curve",
                    "Slurve", "Sweeper", "Slow Curve", "Screwball")
  
offspeed <- c("Changeup", "Split-Finger", "Knuckleball", "Forkball")

fastball_data <- new_statcast %>%
  filter(pitch_name %in% fastballs) %>%
  select(batter_chase, pfx_x, pfx_z, plate_x, plate_z, 
         p_throws, stand, release_speed)

breaking_ball_data <- new_statcast %>%
  filter(pitch_name %in% breaking_balls) %>%
  select(batter_chase, pfx_x, pfx_z, plate_x, plate_z, 
         p_throws, stand, release_speed)

offspeed_data <- new_statcast %>%
  filter(pitch_name %in% offspeed) %>%
  select(batter_chase, pfx_x, pfx_z, plate_x, plate_z, 
         p_throws, stand, release_speed)

set.seed(43254325)
fastball_split <- createDataPartition(fastball_data$batter_chase, 
                                      p = .80, list = FALSE)
fastball_train <- fastball_data[fastball_split, ]
fastball_test <- fastball_data[-fastball_split, ]

set.seed(43254325)
breaking_ball_split <- createDataPartition(breaking_ball_data$batter_chase, 
                                           p = .80, list = FALSE)
breaking_ball_train <- breaking_ball_data[breaking_ball_split, ]
breaking_ball_test <- breaking_ball_data[-breaking_ball_split, ]

set.seed(43254325)
offspeed_split <- createDataPartition(offspeed_data$batter_chase, 
                                      p = .80, list = FALSE)
offspeed_train <- offspeed_data[offspeed_split, ]
offspeed_test <- offspeed_data[-offspeed_split, ]

# Logit Regression
fastball_logit <- glm(batter_chase ~ pfx_x + pfx_z + plate_x +
                        plate_z + p_throws + stand + release_speed,
                  data = fastball_train,
                  family = binomial())

p <- predict(fastball_logit, fastball_test, type = "response")
p <- ifelse(p >= 0.5, 1, 0)

breaking_ball_logit <- glm(batter_chase ~ pfx_x + pfx_z + 
                             plate_x + plate_z + p_throws + 
                             stand + release_speed,
                       data = breaking_ball_train,
                       family = binomial())

offspeed_logit <- glm(batter_chase ~ pfx_x + pfx_z + plate_x +
                        plate_z + p_throws + stand + release_speed,
                      data = offspeed_train,
                      family = binomial())

#mod1_logit_reduced <- glm(batter_chase ~ pfx_x + pfx_z,
                          #data = statcast_train,
                          #family = "binomial")

#anova(mod1_logit_reduced, mod1_logit, test = "LRT")

summary(fastball_logit)
exp(coef(fastball_logit))
vif(fastball_logit)

summary(breaking_ball_logit)
exp(coef(breaking_ball_logit))
vif(breaking_ball_logit)

summary(offspeed_logit)
exp(coef(offspeed_logit))
vif(offspeed_logit)

# Assessing model accuracy
set.seed(34235235)
cv_fastball_logit <- train(
  batter_chase ~ pfx_x + pfx_z + plate_x + plate_z + p_throws +
    stand + release_speed,
  data = fastball_train,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

cv_breaking_ball_logit <- train(
  batter_chase ~ pfx_x + pfx_z + plate_x + plate_z + p_throws +
    stand + release_speed,
  data = breaking_ball_train,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

cv_offspeed_logit <- train(
  batter_chase ~ pfx_x + pfx_z + plate_x + plate_z + p_throws +
    stand + release_speed,
  data = offspeed_train,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

summary(
  resamples(
    list(
      fastball_mod = cv_fastball_logit,
      breaking_ball_mod = cv_breaking_ball_logit,
      offspeed_mod = cv_offspeed_logit
    )
  )
)$statistics$Accuracy

fast_preds <- predict(cv_fastball_logit, newdata = fastball_test)
bb_preds <- predict(cv_breaking_ball_logit, 
                    newdata = breaking_ball_test)
offspeed_preds <- predict(cv_offspeed_logit, 
                          newdata = offspeed_test)

threshold <- 0.5
# Convert probabilities to binary predictions
pred_class <- ifelse(pred_class >= threshold, 1, 0)

# 1 is chase, 0 is no chase but swing
confusionMatrix(
  data = relevel(fast_preds, ref = 1), 
  reference = relevel(fastball_test$batter_chase, ref = 1)
)

confusionMatrix(
  data = relevel(bb_preds, ref = 1), 
  reference = relevel(breaking_ball_test$batter_chase, ref = 1)
)

confusionMatrix(
  data = relevel(offspeed_preds, ref = 1), 
  reference = relevel(offspeed_test$batter_chase, ref = 1)
)

# Compute predicted probabilities
logit_prob <- predict(cv_mod1_logit, statcast_train, type = "prob")$`1`
probit_prob <- predict(cv_mod1_probit, statcast_train, type = "prob")$`1`

# Compute AUC metrics for models
perf1 <- prediction(logit_prob, statcast_train$batter_chase) %>%
  performance(measure = "tpr", x.measure = "fpr")
perf2 <- prediction(probit_prob, statcast_train$chased) %>%
  performance(measure = "tpr", x.measure = "fpr")

# Plot ROC curves for models
plot(perf1, col = "black", lty = 2)
plot(perf2, add = TRUE, col = "blue")
legend(0.8, 0.2, legend = c("mod1_logit", "mod1_probit"),
       col = c("black", "blue"), lty = 2:1, cex = 0.6)

vip(cv_fastball_logit)
