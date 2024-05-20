#### self kd only baseline (method_start=1, =25)
#loss mode + loss weight
#l2 so far this has the better results， yet has slightly better rankme final score, how about trend is still the same [p.s. the sfmx results is not promising]

####just original kd_loss, method start after epoch 1
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |83.90 |85.65 |89.45 |90.30 |89.98 |87.77 |88.09 |89.97 |89.83 |89.76 |
|rankme     |58.88 |↓65.38|↓65.24|↓62.01|↓58.59|↓53.58|↓54.15|↓53.40|↓55.08|↓55.84|
####just original kd_loss, method start after epoch 25
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |79.37 |85.19 |91.26 |91.31 |91.12 |90.20 |89.18 |89.26 |88.37 |88.50 |
|rankme     |↓36.01|↓56.41|↓59.18|↓57.24|↓56.95|↓55.41|↓55.50|↓54.97|↓54.55|↓56.33|
####sfmx is not so good

#kd: better than CS

####just original kd_loss, method start after epoch 1
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |89.65 |89.86 |89.98 |90.21 |89.67 |89.89 |88.55 |89.22 |88.76 |89.66 |
|rankme     |63.78 |↓62.34|↓58.96|↓56.72|↓57.38|↓56.21|↓56.94|↓56.72|↓56.68|↓56.76|
####just original kd_loss, method start after epoch 25 (comparing with the epoch 1 version, the rankme peak is not so high but demonstrates less performance especially on smaller loss_weight)
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |90.01 |89.64 |89.84 |90.00 |89.42 |88.83 |90.00 |86.79 |86.73 |87.87 |
|rankme     |↓63.06|↓60.90|↓58.17|↓57.75|↓57.99|↓57.95|↓57.05|↓57.50|↓57.00|↓57.58|

#cs: not so good
# similar to kd results
############ Need to summarize this results?
1.l2 so far has performed more closely to pure supervised results
2.rankme do not show good correlation, but a bad rankme surely leads to bad accuracy
3.

#### to run kd ema and (method_start=25, also=1 ?)
### test loss_weight and ema ratio respectively
######################################## l2 ######################################## 
sfmx_25
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |90.64 |90.51 |91.01 |90.04 |90.28 |90.31 |91.11 |90.10 |90.09 |90.58 |
|rankme     |↓59.53|↓59.83|↓57.00|↓56.94|↓56.45|↓57.89|↓57.57|↓57.40|↓57.34|↓57.10| 62/60
                                    sfmx 🛫
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |90.37 |90.48 |90.18 |90.63 |90.80 |91.01 |90.48 |90.95 |90.71 |90.52 |
|rankme     |↓57.95|↓57.76|↓57.55|↓56.73|↓57.00|↓56.79|↓57.30|↓59.88|↓57.04|↓56.73| 62/60
                                    _25
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |xxxxx |90.41 |92.01 |91.99 |92.11 |91.58 |91.15 |90.96 |90.72 |90.51 |
|rankme     |↓xxxxx|↓50.05|↓58.18|↓57.24|↓57.02|↓55.20|↓55.94|↓56.50|↓55.65|↓56.63| 62/60
                                    _1
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |80.15 |80.51 |86.47 |88.64 |89.81 |91.51 |91.28 |91.51 |91.42 |91.72 |
|rankme     |↓58.74|↓61.40|↓60.78|↓66.67|↓67.07|↓61.91|↓62.40|↓59.88|↓58.12|↓55.21| 62/60

######################################## cs ######################################## 
                                    sfmx_25
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |90.02 |90.51 |89.69 |90.73 |90.67 |89.69 |90.58 |89.59 |90.26 |89.30 |
|rankme     |↓69.85|↓65.19|↓58.43|↓57.57|↓58.05|↓57.32|↓57.89|↓56.55|↓56.89|↓57.02| 62/60
sfmx

                                    _25
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |91.53 |91.60 |90.20 |90.27 |90.75 |90.87 |90.19 |90.04 |90.14 |90.71 |
|rankme     |↓57.15|↓56.51|↓55.78|↓56.64|↓56.36|↓57.78|↓57.49|↓55.88|↓56.76|↓56.76| 62/60
                                    _1
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |89.44 |89.57 |90.94 |91.01 |91.19 |91.21 |91.33 |90.97 |90.82 |91.41 |
|rankme     |↓63.11|↓61.40|↓56.75|↓56.18|↓56.36|↓56.65|↓55.76|↓56.64|↓56.28|↓55.06| 62/60
########################################  kd ########################################  wow works really well, seemingly !
                                    sfmx_25 # 
|loss_weight|15.0  |10.0  |7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |91.50 |91.42 |91.58 |91.49 |91.07 |90.80 |91.40 |90.13 |90.23 |90.52 |89.98 |90.27 |
|rankme     |76.91 |71.09 |67.47 |63.60 |↓58.34|↓57.87|↓57.21|↓57.97|↓58.06|↓56.54|↓55.73|↓56.98| 62/60
                                    _25 ✅ add 15.0 10.0 2.0ema 
|loss_weight|7.0   |5.0   |1.0   |0.5   |0.3   |0.1   |0.09  |0.05  |0.03  |0.01  |
|accuracy   |91.92 |91.70 |91.05 |90.30 |90.36 |90.13 |90.89 |90.80 |90.15 |90.67 |
|rankme     |↓66.09|↓63.36|↓58.22|↓56.65|↓57.81|↓57.97|↓57.4 6|↓57.40|↓56.98|↓57.48| 62/60
_1

######################################### kd baseline ema ratio ######################
_25
#not really good
#only <0.1 ratio reach better performance, but with low weight

_sfmx25



######################################## kd with classifier-only
######kd start epoch =1, 5, 25, 30? [5 epoch freeze?]
##+ [1epoch freeze]?no use cuz ema: >1 >24 wins
#######- now only tested on ema self-kd results, the lower ema ratio is performing better 
some setting achieve slightly better results
see if plot makes it more clearer?

######l2 cs try only the above best option

######kd ema try only the above best option
<!-- ############### MASK THEN ############### -->
######kd all mode