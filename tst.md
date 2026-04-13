[info] Using device: cpu

======================================================================
  LOADING DATA
======================================================================
Shape : (14980, 15)
Target distribution:
eyeDetection
0    8257
1    6723
Name: count, dtype: int64

======================================================================
  HOLD-OUT SPLIT  70/15/15  (train=70%  cv=15%  test=15%)
======================================================================
  Train:10486  CV:2247  Test:2247

  ── Classical ML ──

  LogisticRegression
  [LogisticRegression | CV]  Acc=0.4250  F1=0.5965  AUC=0.5720
  [LogisticRegression | Test]  Acc=0.2408  F1=0.1629  AUC=0.6233
    Train time: 0.1s

  SVM_RBF
  [SVM_RBF | CV]  Acc=0.4611  F1=0.5846  AUC=0.5494
  [SVM_RBF | Test]  Acc=0.3035  F1=0.1213  AUC=0.4349
    Train time: 38.7s

  RandomForest
  [RandomForest | CV]  Acc=0.5038  F1=0.5716  AUC=0.6042
  [RandomForest | Test]  Acc=0.5305  F1=0.1471  AUC=0.4693
    Train time: 1.2s

  GradientBoosting
  [GradientBoosting | CV]  Acc=0.4909  F1=0.5763  AUC=0.6154
  [GradientBoosting | Test]  Acc=0.5456  F1=0.1442  AUC=0.5195
    Train time: 13.9s

  XGBoost
  [XGBoost | CV]  Acc=0.5149  F1=0.6036  AUC=0.6867
  [XGBoost | Test]  Acc=0.5745  F1=0.1744  AUC=0.5613
    Train time: 0.7s

  ── Deep Learning ──

  LSTM
    Epoch   5/20  loss=0.7420  cv_f1=0.6205
    Epoch  10/20  loss=0.7948  cv_f1=0.6100
    Epoch  15/20  loss=0.7566  cv_f1=0.5931
    Epoch  20/20  loss=0.7030  cv_f1=0.5909
  [LSTM | CV]  Acc=0.4530  F1=0.6205  AUC=0.5491
  [LSTM | Test]  Acc=0.1961  F1=0.1418  AUC=0.7910

  CNN_LSTM
    Epoch   5/20  loss=0.6901  cv_f1=0.0540
    Epoch  10/20  loss=0.5436  cv_f1=0.4510
    Epoch  15/20  loss=0.3471  cv_f1=0.4323
    Epoch  20/20  loss=0.2749  cv_f1=0.3595
  [CNN_LSTM | CV]  Acc=0.4993  F1=0.4510  AUC=0.5771
  [CNN_LSTM | Test]  Acc=0.6088  F1=0.1460  AUC=0.6208

  EEGTransformer
    Epoch   5/20  loss=0.9033  cv_f1=0.6199
    Epoch  10/20  loss=0.8580  cv_f1=0.6518
    Epoch  15/20  loss=0.8657  cv_f1=0.6563
    Epoch  20/20  loss=0.8527  cv_f1=0.6541
  [EEGTransformer | CV]  Acc=0.5529  F1=0.6563  AUC=0.5660
  [EEGTransformer | Test]  Acc=0.3078  F1=0.1563  AUC=0.6889

======================================================================
  HOLD-OUT SPLIT  60/20/20  (train=60%  cv=20%  test=20%)
======================================================================
  Train:8988  CV:2996  Test:2996

  ── Classical ML ──

  LogisticRegression
  [LogisticRegression | CV]  Acc=0.3241  F1=0.4725  AUC=0.5906
  [LogisticRegression | Test]  Acc=0.1252  F1=0.1755  AUC=0.6320
    Train time: 0.0s

  SVM_RBF
  [SVM_RBF | CV]  Acc=0.2814  F1=0.4116  AUC=0.2704
  [SVM_RBF | Test]  Acc=0.2864  F1=0.1674  AUC=0.5751
    Train time: 16.5s

  RandomForest
  [RandomForest | CV]  Acc=0.3124  F1=0.4128  AUC=0.3388
  [RandomForest | Test]  Acc=0.4092  F1=0.1828  AUC=0.5646
    Train time: 1.0s

  GradientBoosting
  [GradientBoosting | CV]  Acc=0.3415  F1=0.4524  AUC=0.4757
  [GradientBoosting | Test]  Acc=0.4169  F1=0.1817  AUC=0.6000
    Train time: 12.1s

  XGBoost
  [XGBoost | CV]  Acc=0.3505  F1=0.4692  AUC=0.4975
  [XGBoost | Test]  Acc=0.4379  F1=0.1966  AUC=0.6440
    Train time: 0.4s

  ── Deep Learning ──

  LSTM
    Epoch   5/20  loss=0.6241  cv_f1=0.4621
    Epoch  10/20  loss=0.6134  cv_f1=0.4621
    Epoch  15/20  loss=0.6087  cv_f1=0.4621
    Epoch  20/20  loss=0.6074  cv_f1=0.4621
  [LSTM | CV]  Acc=0.3005  F1=0.4621  AUC=0.6717
  [LSTM | Test]  Acc=0.0737  F1=0.1372  AUC=0.6224

  CNN_LSTM
    Epoch   5/20  loss=0.6832  cv_f1=0.4621
    Epoch  10/20  loss=0.4426  cv_f1=0.4929
    Epoch  15/20  loss=0.1991  cv_f1=0.4166
    Epoch  20/20  loss=0.0865  cv_f1=0.4444
  [CNN_LSTM | CV]  Acc=0.3817  F1=0.4929  AUC=0.4990
  [CNN_LSTM | Test]  Acc=0.0747  F1=0.1374  AUC=0.3600

  EEGTransformer
    Epoch   5/20  loss=0.6707  cv_f1=0.4621
    Epoch  10/20  loss=0.8113  cv_f1=0.4621
    Epoch  15/20  loss=0.7665  cv_f1=0.3990
    Epoch  20/20  loss=0.5751  cv_f1=0.2882
  [EEGTransformer | CV]  Acc=0.3005  F1=0.4621  AUC=0.7001
  [EEGTransformer | Test]  Acc=0.0737  F1=0.1372  AUC=0.8138

======================================================================
  HOLD-OUT SPLIT  80/10/10  (train=80%  cv=10%  test=10%)
======================================================================
  Train:11984  CV:1498  Test:1498

  ── Classical ML ──

  LogisticRegression
  [LogisticRegression | CV]  Acc=0.1462  F1=0.2080  AUC=0.3485
  [LogisticRegression | Test]  Acc=0.4566  F1=0.1133  AUC=0.4437
    Train time: 0.1s

  SVM_RBF
  [SVM_RBF | CV]  Acc=0.1889  F1=0.2033  AUC=0.5658
  [SVM_RBF | Test]  Acc=0.1776  F1=0.0861  AUC=0.3549
    Train time: 54.6s

  RandomForest
  [RandomForest | CV]  Acc=0.4980  F1=0.2613  AUC=0.6670
  [RandomForest | Test]  Acc=0.4866  F1=0.0964  AUC=0.3696
    Train time: 1.3s

  GradientBoosting
  [GradientBoosting | CV]  Acc=0.5087  F1=0.2698  AUC=0.6721
  [GradientBoosting | Test]  Acc=0.4666  F1=0.0827  AUC=0.3369
    Train time: 15.9s

  XGBoost
  [XGBoost | CV]  Acc=0.4760  F1=0.2752  AUC=0.6965
  [XGBoost | Test]  Acc=0.4987  F1=0.0941  AUC=0.3951
    Train time: 0.5s

  ── Deep Learning ──

  LSTM
    Epoch   5/20  loss=0.8398  cv_f1=0.1047
    Epoch  10/20  loss=0.7034  cv_f1=0.1580
    Epoch  15/20  loss=0.6669  cv_f1=0.1965
    Epoch  20/20  loss=0.6177  cv_f1=0.1939
  [LSTM | CV]  Acc=0.2985  F1=0.1965  AUC=0.8666
  [LSTM | Test]  Acc=0.3431  F1=0.1649  AUC=0.7184

  CNN_LSTM
    Epoch   5/20  loss=0.7543  cv_f1=0.2553
    Epoch  10/20  loss=0.6210  cv_f1=0.3548
    Epoch  15/20  loss=0.4545  cv_f1=0.3137
    Epoch  20/20  loss=0.3434  cv_f1=0.4549
  [CNN_LSTM | CV]  Acc=0.9114  F1=0.4549  AUC=0.7913
  [CNN_LSTM | Test]  Acc=0.8250  F1=0.1068  AUC=0.6949

  EEGTransformer
    Epoch   5/20  loss=0.6976  cv_f1=0.1580
    Epoch  10/20  loss=0.6925  cv_f1=0.1580
    Epoch  15/20  loss=0.6909  cv_f1=0.1580
    Epoch  20/20  loss=0.6899  cv_f1=0.1580
  [EEGTransformer | CV]  Acc=0.0858  F1=0.1580  AUC=0.1946
  [EEGTransformer | Test]  Acc=0.0649  F1=0.1218  AUC=0.2917

======================================================================
  TEMPORAL CROSS-VALIDATION — Walk-Forward (Expanding Window)
======================================================================

  Fold 1  train=7490  val=1248
  [LogisticRegression|fold1]  Acc=0.8141  F1=0.8975  AUC=nan
  [SVM_RBF|fold1]  Acc=0.7364  F1=0.8482  AUC=nan
  [RandomForest|fold1]  Acc=0.7532  F1=0.8592  AUC=nan
  [GradientBoosting|fold1]  Acc=0.7724  F1=0.8716  AUC=nan
  [XGBoost|fold1]  Acc=0.7700  F1=0.8701  AUC=nan

  Fold 2  train=8738  val=1248
  [LogisticRegression|fold2]  Acc=0.2620  F1=0.3417  AUC=0.1755
  [SVM_RBF|fold2]  Acc=0.2460  F1=0.3433  AUC=0.2632
  [RandomForest|fold2]  Acc=0.3037  F1=0.4027  AUC=0.5037
  [GradientBoosting|fold2]  Acc=0.2997  F1=0.3947  AUC=0.4000
  [XGBoost|fold2]  Acc=0.2812  F1=0.3827  AUC=0.3799

  Fold 3  train=9986  val=1248
  [LogisticRegression|fold3]  Acc=0.1106  F1=0.1886  AUC=0.8557
  [SVM_RBF|fold3]  Acc=0.2035  F1=0.1548  AUC=0.4812
  [RandomForest|fold3]  Acc=0.3526  F1=0.2186  AUC=0.6451
  [GradientBoosting|fold3]  Acc=0.3405  F1=0.2184  AUC=0.6676
  [XGBoost|fold3]  Acc=0.3397  F1=0.2167  AUC=0.7287

  Fold 4  train=11234  val=1248
  [LogisticRegression|fold4]  Acc=0.3806  F1=0.4673  AUC=0.3331
  [SVM_RBF|fold4]  Acc=0.3974  F1=0.4659  AUC=0.3480
  [RandomForest|fold4]  Acc=0.3750  F1=0.3819  AUC=0.3377
  [GradientBoosting|fold4]  Acc=0.4135  F1=0.4610  AUC=0.3634
  [XGBoost|fold4]  Acc=0.4455  F1=0.4758  AUC=0.4035

  Fold 5  train=12482  val=1248
  [LogisticRegression|fold5]  Acc=0.1763  F1=0.1518  AUC=0.6040
  [SVM_RBF|fold5]  Acc=0.2620  F1=0.1203  AUC=0.4602
  [RandomForest|fold5]  Acc=0.5417  F1=0.1006  AUC=0.4308
  [GradientBoosting|fold5]  Acc=0.5329  F1=0.0876  AUC=0.4331
  [XGBoost|fold5]  Acc=0.5441  F1=0.1039  AUC=0.4444

  Walk-Forward CV — Mean ± Std across folds
  LogisticRegression              Acc=0.3487±0.2496  F1=0.4094±0.2688  AUC=nan±nan
  SVM_RBF                         Acc=0.3691±0.1948  F1=0.3865±0.2630  AUC=nan±nan
  RandomForest                    Acc=0.4652±0.1647  F1=0.3926±0.2583  AUC=nan±nan
  GradientBoosting                Acc=0.4718±0.1699  F1=0.4067±0.2670  AUC=nan±nan
  XGBoost                         Acc=0.4761±0.1723  F1=0.4098±0.2637  AUC=nan±nan

======================================================================
  TEMPORAL CROSS-VALIDATION — Sliding Window
======================================================================

  Fold 1  train=7490  val=1248
  [LogisticRegression|fold1]  Acc=0.8141  F1=0.8975  AUC=nan
  [SVM_RBF|fold1]  Acc=0.7364  F1=0.8482  AUC=nan
  [RandomForest|fold1]  Acc=0.7532  F1=0.8592  AUC=nan
  [GradientBoosting|fold1]  Acc=0.7724  F1=0.8716  AUC=nan
  [XGBoost|fold1]  Acc=0.7700  F1=0.8701  AUC=nan

  Fold 2  train=7490  val=1248
  [LogisticRegression|fold2]  Acc=0.2420  F1=0.3582  AUC=0.2213
  [SVM_RBF|fold2]  Acc=0.2644  F1=0.3616  AUC=0.3343
  [RandomForest|fold2]  Acc=0.2829  F1=0.3899  AUC=0.4794
  [GradientBoosting|fold2]  Acc=0.2869  F1=0.3862  AUC=0.3569
  [XGBoost|fold2]  Acc=0.2724  F1=0.3668  AUC=0.3541

  Fold 3  train=7490  val=1248
  [LogisticRegression|fold3]  Acc=0.5865  F1=0.2773  AUC=0.7193
  [SVM_RBF|fold3]  Acc=0.5152  F1=0.2789  AUC=0.7490
  [RandomForest|fold3]  Acc=0.4712  F1=0.2272  AUC=0.6646
  [GradientBoosting|fold3]  Acc=0.4559  F1=0.2362  AUC=0.7482
  [XGBoost|fold3]  Acc=0.4655  F1=0.2412  AUC=0.7846

  Fold 4  train=7490  val=1248
  [LogisticRegression|fold4]  Acc=0.3694  F1=0.2337  AUC=0.4458
  [SVM_RBF|fold4]  Acc=0.4111  F1=0.3213  AUC=0.6079
  [RandomForest|fold4]  Acc=0.4287  F1=0.4399  AUC=0.3678
  [GradientBoosting|fold4]  Acc=0.4319  F1=0.4813  AUC=0.3885
  [XGBoost|fold4]  Acc=0.4784  F1=0.5124  AUC=0.4603

  Fold 5  train=7490  val=1248
  [LogisticRegression|fold5]  Acc=0.3870  F1=0.1603  AUC=0.6026
  [SVM_RBF|fold5]  Acc=0.3317  F1=0.1184  AUC=0.4726
  [RandomForest|fold5]  Acc=0.5537  F1=0.0854  AUC=0.4122
  [GradientBoosting|fold5]  Acc=0.5321  F1=0.0960  AUC=0.4012
  [XGBoost|fold5]  Acc=0.5473  F1=0.1102  AUC=0.4296

  Sliding-Window CV — Mean ± Std across folds
  LogisticRegression              Acc=0.4798±0.2003  F1=0.3854±0.2640  AUC=nan±nan
  SVM_RBF                         Acc=0.4518±0.1650  F1=0.3857±0.2455  AUC=nan±nan
  RandomForest                    Acc=0.4979±0.1549  F1=0.4003±0.2613  AUC=nan±nan
  GradientBoosting                Acc=0.4958±0.1595  F1=0.4143±0.2636  AUC=nan±nan
  XGBoost                         Acc=0.5067±0.1602  F1=0.4201±0.2615  AUC=nan±nan

======================================================================
  FINAL SUMMARY — Hold-Out Splits (Test partition)
======================================================================
  Model                           Split            Acc        F1       AUC
  -----------------------------------------------------------------
  LogisticRegression              70/15/15      0.2408    0.1629    0.6233
  SVM_RBF                         70/15/15      0.3035    0.1213    0.4349
  RandomForest                    70/15/15      0.5305    0.1471    0.4693
  GradientBoosting                70/15/15      0.5456    0.1442    0.5195
  XGBoost                         70/15/15      0.5745    0.1744    0.5613
  LSTM                            70/15/15      0.1961    0.1418    0.7910
  CNN_LSTM                        70/15/15      0.6088    0.1460    0.6208
  EEGTransformer                  70/15/15      0.3078    0.1563    0.6889
  LogisticRegression              60/20/20      0.1252    0.1755    0.6320
  SVM_RBF                         60/20/20      0.2864    0.1674    0.5751
  RandomForest                    60/20/20      0.4092    0.1828    0.5646
  GradientBoosting                60/20/20      0.4169    0.1817    0.6000
  XGBoost                         60/20/20      0.4379    0.1966    0.6440
  LSTM                            60/20/20      0.0737    0.1372    0.6224
  CNN_LSTM                        60/20/20      0.0747    0.1374    0.3600
  EEGTransformer                  60/20/20      0.0737    0.1372    0.8138
  LogisticRegression              80/10/10      0.4566    0.1133    0.4437
  SVM_RBF                         80/10/10      0.1776    0.0861    0.3549
  RandomForest                    80/10/10      0.4866    0.0964    0.3696
  GradientBoosting                80/10/10      0.4666    0.0827    0.3369
  XGBoost                         80/10/10      0.4987    0.0941    0.3951
  LSTM                            80/10/10      0.3431    0.1649    0.7184
  CNN_LSTM                        80/10/10      0.8250    0.1068    0.6949
  EEGTransformer                  80/10/10      0.0649    0.1218    0.2917