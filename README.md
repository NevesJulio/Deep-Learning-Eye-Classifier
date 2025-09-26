# Deep-Learning-Eye-Classifier


Treinamento RESNET50 pré-treinada

```
Epoch 1/10 | Train Loss: 0.1388 Acc: 0.9418 | Val Loss: 0.0566 Acc: 0.9862
Epoch 2/10 | Train Loss: 0.0298 Acc: 0.9891 | Val Loss: 0.0593 Acc: 0.9803
Epoch 3/10 | Train Loss: 0.0160 Acc: 0.9965 | Val Loss: 0.0422 Acc: 0.9822
Epoch 4/10 | Train Loss: 0.0038 Acc: 1.0000 | Val Loss: 0.0454 Acc: 0.9822
Epoch 5/10 | Train Loss: 0.0035 Acc: 1.0000 | Val Loss: 0.0447 Acc: 0.9822
Epoch 6/10 | Train Loss: 0.0064 Acc: 0.9985 | Val Loss: 0.0563 Acc: 0.9842
Epoch 7/10 | Train Loss: 0.0325 Acc: 0.9896 | Val Loss: 0.0604 Acc: 0.9862
Epoch 8/10 | Train Loss: 0.0254 Acc: 0.9921 | Val Loss: 0.0437 Acc: 0.9822
Epoch 9/10 | Train Loss: 0.0127 Acc: 0.9965 | Val Loss: 0.0695 Acc: 0.9822
Epoch 10/10 | Train Loss: 0.0032 Acc: 1.0000 | Val Loss: 0.0504 Acc: 0.9862
```

Treinamento DENSENET121 pré-treinada

```
Epoch 1/10 | Train Loss: 0.1683 Acc: 0.9348 | Val Loss: 0.0660 Acc: 0.9783
Epoch 2/10 | Train Loss: 0.0441 Acc: 0.9882 | Val Loss: 0.0510 Acc: 0.9803
Epoch 3/10 | Train Loss: 0.0109 Acc: 0.9995 | Val Loss: 0.0302 Acc: 0.9941
Epoch 4/10 | Train Loss: 0.0045 Acc: 1.0000 | Val Loss: 0.0308 Acc: 0.9882
Epoch 5/10 | Train Loss: 0.0049 Acc: 0.9990 | Val Loss: 0.0368 Acc: 0.9882
Epoch 6/10 | Train Loss: 0.0096 Acc: 0.9970 | Val Loss: 0.0393 Acc: 0.9901
Epoch 7/10 | Train Loss: 0.0303 Acc: 0.9906 | Val Loss: 0.0871 Acc: 0.9724
Epoch 8/10 | Train Loss: 0.0193 Acc: 0.9951 | Val Loss: 0.0800 Acc: 0.9724
Epoch 9/10 | Train Loss: 0.0152 Acc: 0.9951 | Val Loss: 0.0389 Acc: 0.9862
Epoch 10/10 | Train Loss: 0.0107 Acc: 0.9961 | Val Loss: 0.0270 Acc: 0.9862
Modelo salvo em densenet121_finetuned.pth
```

Treinamento VisualTransformer pré-treinada

```
Epoch 1/10 | Train Loss: 0.2210 Acc: 0.9038 | Val Loss: 0.0879 Acc: 0.9684
Epoch 2/10 | Train Loss: 0.1024 Acc: 0.9605 | Val Loss: 0.0721 Acc: 0.9724
Epoch 3/10 | Train Loss: 0.0455 Acc: 0.9798 | Val Loss: 0.1234 Acc: 0.9625
Epoch 4/10 | Train Loss: 0.0413 Acc: 0.9872 | Val Loss: 0.0596 Acc: 0.9704
Epoch 5/10 | Train Loss: 0.0449 Acc: 0.9812 | Val Loss: 0.0595 Acc: 0.9763
Epoch 6/10 | Train Loss: 0.0181 Acc: 0.9931 | Val Loss: 0.0429 Acc: 0.9803
Epoch 7/10 | Train Loss: 0.0498 Acc: 0.9803 | Val Loss: 0.0667 Acc: 0.9684
Epoch 8/10 | Train Loss: 0.0403 Acc: 0.9837 | Val Loss: 0.0716 Acc: 0.9724
Epoch 9/10 | Train Loss: 0.0276 Acc: 0.9891 | Val Loss: 0.0922 Acc: 0.9744
Epoch 10/10 | Train Loss: 0.0392 Acc: 0.9877 | Val Loss: 0.0749 Acc: 0.9704
Modelo salvo em VisualTransformer_finetuned.pth
```



Treinamento teste da ResNet50 por 5 repetições.

```
===== Run 1/5 =====
Epoch 1/10 | Train Loss: 0.1377 Acc: 0.9472 | Val Loss: 0.0801 Acc: 0.9625 | Prec: 0.9950 Rec: 0.9171 F1: 0.9544 ROC_AUC: 0.9988
Epoch 2/10 | Train Loss: 0.0326 Acc: 0.9886 | Val Loss: 0.0671 Acc: 0.9744 | Prec: 0.9554 Rec: 0.9862 F1: 0.9705 ROC_AUC: 0.9977
Epoch 3/10 | Train Loss: 0.0162 Acc: 0.9941 | Val Loss: 0.0716 Acc: 0.9763 | Prec: 0.9638 Rec: 0.9816 F1: 0.9726 ROC_AUC: 0.9952
Epoch 4/10 | Train Loss: 0.0136 Acc: 0.9961 | Val Loss: 0.0495 Acc: 0.9842 | Prec: 0.9772 Rec: 0.9862 F1: 0.9817 ROC_AUC: 0.9991
Epoch 5/10 | Train Loss: 0.0100 Acc: 0.9970 | Val Loss: 0.0549 Acc: 0.9803 | Prec: 0.9559 Rec: 1.0000 F1: 0.9775 ROC_AUC: 0.9992
Epoch 6/10 | Train Loss: 0.0035 Acc: 0.9995 | Val Loss: 0.0275 Acc: 0.9901 | Prec: 0.9818 Rec: 0.9954 F1: 0.9886 ROC_AUC: 0.9996
Epoch 7/10 | Train Loss: 0.0023 Acc: 0.9990 | Val Loss: 0.0481 Acc: 0.9822 | Prec: 0.9906 Rec: 0.9677 F1: 0.9790 ROC_AUC: 0.9988
Epoch 8/10 | Train Loss: 0.0019 Acc: 0.9995 | Val Loss: 0.0444 Acc: 0.9862 | Prec: 0.9773 Rec: 0.9908 F1: 0.9840 ROC_AUC: 0.9994
Epoch 9/10 | Train Loss: 0.0180 Acc: 0.9946 | Val Loss: 0.0322 Acc: 0.9862 | Prec: 0.9688 Rec: 1.0000 F1: 0.9841 ROC_AUC: 0.9996
Epoch 10/10 | Train Loss: 0.0106 Acc: 0.9980 | Val Loss: 0.0506 Acc: 0.9803 | Prec: 0.9641 Rec: 0.9908 F1: 0.9773 ROC_AUC: 0.9990


===== Run 2/5 =====
Epoch 1/10 | Train Loss: 0.1269 Acc: 0.9487 | Val Loss: 0.0398 Acc: 0.9822 | Prec: 0.9906 Rec: 0.9677 F1: 0.9790 ROC_AUC: 0.9994
Epoch 2/10 | Train Loss: 0.0492 Acc: 0.9812 | Val Loss: 0.0527 Acc: 0.9744 | Prec: 0.9679 Rec: 0.9724 F1: 0.9701 ROC_AUC: 0.9982
Epoch 3/10 | Train Loss: 0.0241 Acc: 0.9926 | Val Loss: 0.0453 Acc: 0.9842 | Prec: 0.9686 Rec: 0.9954 F1: 0.9818 ROC_AUC: 0.9990
Epoch 4/10 | Train Loss: 0.0166 Acc: 0.9956 | Val Loss: 0.0265 Acc: 0.9921 | Prec: 0.9863 Rec: 0.9954 F1: 0.9908 ROC_AUC: 0.9995
Epoch 5/10 | Train Loss: 0.0039 Acc: 0.9995 | Val Loss: 0.0420 Acc: 0.9862 | Prec: 0.9730 Rec: 0.9954 F1: 0.9841 ROC_AUC: 0.9992
Epoch 6/10 | Train Loss: 0.0014 Acc: 1.0000 | Val Loss: 0.0347 Acc: 0.9921 | Prec: 0.9863 Rec: 0.9954 F1: 0.9908 ROC_AUC: 0.9973
Epoch 7/10 | Train Loss: 0.0058 Acc: 0.9985 | Val Loss: 0.0468 Acc: 0.9822 | Prec: 0.9771 Rec: 0.9816 F1: 0.9793 ROC_AUC: 0.9991
Epoch 8/10 | Train Loss: 0.0025 Acc: 0.9995 | Val Loss: 0.0319 Acc: 0.9901 | Prec: 0.9818 Rec: 0.9954 F1: 0.9886 ROC_AUC: 0.9994
Epoch 9/10 | Train Loss: 0.0008 Acc: 1.0000 | Val Loss: 0.0408 Acc: 0.9862 | Prec: 0.9730 Rec: 0.9954 F1: 0.9841 ROC_AUC: 0.9995
Epoch 10/10 | Train Loss: 0.0002 Acc: 1.0000 | Val Loss: 0.0318 Acc: 0.9901 | Prec: 0.9818 Rec: 0.9954 F1: 0.9886 ROC_AUC: 0.9995


===== Run 3/5 =====
Epoch 1/10 | Train Loss: 0.1408 Acc: 0.9388 | Val Loss: 0.2725 Acc: 0.9448 | Prec: 1.0000 Rec: 0.8710 F1: 0.9310 ROC_AUC: 0.9816
Epoch 2/10 | Train Loss: 0.0369 Acc: 0.9891 | Val Loss: 0.0525 Acc: 0.9882 | Prec: 1.0000 Rec: 0.9724 F1: 0.9860 ROC_AUC: 0.9976
Epoch 3/10 | Train Loss: 0.0141 Acc: 0.9956 | Val Loss: 0.0882 Acc: 0.9704 | Prec: 0.9391 Rec: 0.9954 F1: 0.9664 ROC_AUC: 0.9988
Epoch 4/10 | Train Loss: 0.0101 Acc: 0.9965 | Val Loss: 0.0461 Acc: 0.9901 | Prec: 0.9907 Rec: 0.9862 F1: 0.9885 ROC_AUC: 0.9987
Epoch 5/10 | Train Loss: 0.0065 Acc: 0.9985 | Val Loss: 0.0446 Acc: 0.9803 | Prec: 0.9641 Rec: 0.9908 F1: 0.9773 ROC_AUC: 0.9992
Epoch 6/10 | Train Loss: 0.0011 Acc: 1.0000 | Val Loss: 0.0355 Acc: 0.9901 | Prec: 0.9953 Rec: 0.9816 F1: 0.9884 ROC_AUC: 0.9993
Epoch 7/10 | Train Loss: 0.0004 Acc: 1.0000 | Val Loss: 0.0339 Acc: 0.9901 | Prec: 0.9907 Rec: 0.9862 F1: 0.9885 ROC_AUC: 0.9990
Epoch 8/10 | Train Loss: 0.0002 Acc: 1.0000 | Val Loss: 0.0370 Acc: 0.9901 | Prec: 0.9907 Rec: 0.9862 F1: 0.9885 ROC_AUC: 0.9987
Epoch 9/10 | Train Loss: 0.0002 Acc: 1.0000 | Val Loss: 0.0363 Acc: 0.9901 | Prec: 0.9907 Rec: 0.9862 F1: 0.9885 ROC_AUC: 0.9990
Epoch 10/10 | Train Loss: 0.0002 Acc: 1.0000 | Val Loss: 0.0375 Acc: 0.9901 | Prec: 0.9907 Rec: 0.9862 F1: 0.9885 ROC_AUC: 0.9991


===== Run 4/5 =====
Epoch 1/10 | Train Loss: 0.1576 Acc: 0.9319 | Val Loss: 0.0568 Acc: 0.9763 | Prec: 0.9767 Rec: 0.9677 F1: 0.9722 ROC_AUC: 0.9982
Epoch 2/10 | Train Loss: 0.0262 Acc: 0.9931 | Val Loss: 0.0309 Acc: 0.9862 | Prec: 0.9861 Rec: 0.9816 F1: 0.9838 ROC_AUC: 0.9995
Epoch 3/10 | Train Loss: 0.0190 Acc: 0.9941 | Val Loss: 0.0388 Acc: 0.9842 | Prec: 0.9772 Rec: 0.9862 F1: 0.9817 ROC_AUC: 0.9994
Epoch 4/10 | Train Loss: 0.0078 Acc: 0.9980 | Val Loss: 0.0540 Acc: 0.9822 | Prec: 1.0000 Rec: 0.9585 F1: 0.9788 ROC_AUC: 0.9990
Epoch 5/10 | Train Loss: 0.0029 Acc: 0.9995 | Val Loss: 0.0313 Acc: 0.9862 | Prec: 0.9817 Rec: 0.9862 F1: 0.9839 ROC_AUC: 0.9995
Epoch 6/10 | Train Loss: 0.0026 Acc: 0.9990 | Val Loss: 0.0223 Acc: 0.9901 | Prec: 0.9818 Rec: 0.9954 F1: 0.9886 ROC_AUC: 0.9999
Epoch 7/10 | Train Loss: 0.0023 Acc: 0.9990 | Val Loss: 0.0276 Acc: 0.9941 | Prec: 1.0000 Rec: 0.9862 F1: 0.9930 ROC_AUC: 0.9993
Epoch 8/10 | Train Loss: 0.0045 Acc: 0.9990 | Val Loss: 0.0529 Acc: 0.9862 | Prec: 0.9773 Rec: 0.9908 F1: 0.9840 ROC_AUC: 0.9982
Epoch 9/10 | Train Loss: 0.0163 Acc: 0.9970 | Val Loss: 0.1934 Acc: 0.9744 | Prec: 1.0000 Rec: 0.9401 F1: 0.9691 ROC_AUC: 0.9962
Epoch 10/10 | Train Loss: 0.0138 Acc: 0.9956 | Val Loss: 0.0289 Acc: 0.9921 | Prec: 0.9908 Rec: 0.9908 F1: 0.9908 ROC_AUC: 0.9995


===== Run 5/5 =====
Epoch 1/10 | Train Loss: 0.1508 Acc: 0.9383 | Val Loss: 0.0468 Acc: 0.9842 | Prec: 0.9953 Rec: 0.9677 F1: 0.9813 ROC_AUC: 0.9993
Epoch 2/10 | Train Loss: 0.0347 Acc: 0.9882 | Val Loss: 0.0381 Acc: 0.9822 | Prec: 0.9685 Rec: 0.9908 F1: 0.9795 ROC_AUC: 0.9995
Epoch 3/10 | Train Loss: 0.0259 Acc: 0.9936 | Val Loss: 0.0259 Acc: 0.9882 | Prec: 0.9862 Rec: 0.9862 F1: 0.9862 ROC_AUC: 0.9997
Epoch 4/10 | Train Loss: 0.0064 Acc: 0.9990 | Val Loss: 0.0413 Acc: 0.9842 | Prec: 0.9686 Rec: 0.9954 F1: 0.9818 ROC_AUC: 0.9997
Epoch 5/10 | Train Loss: 0.0034 Acc: 0.9985 | Val Loss: 0.0221 Acc: 0.9901 | Prec: 0.9862 Rec: 0.9908 F1: 0.9885 ROC_AUC: 0.9997
Epoch 6/10 | Train Loss: 0.0042 Acc: 0.9985 | Val Loss: 0.0322 Acc: 0.9882 | Prec: 0.9953 Rec: 0.9770 F1: 0.9860 ROC_AUC: 0.9992
Epoch 7/10 | Train Loss: 0.0040 Acc: 0.9980 | Val Loss: 0.0819 Acc: 0.9763 | Prec: 1.0000 Rec: 0.9447 F1: 0.9716 ROC_AUC: 0.9981
Epoch 8/10 | Train Loss: 0.0033 Acc: 1.0000 | Val Loss: 0.0595 Acc: 0.9803 | Prec: 0.9726 Rec: 0.9816 F1: 0.9771 ROC_AUC: 0.9985
Epoch 9/10 | Train Loss: 0.0090 Acc: 0.9970 | Val Loss: 0.0670 Acc: 0.9783 | Prec: 0.9725 Rec: 0.9770 F1: 0.9747 ROC_AUC: 0.9982
Epoch 10/10 | Train Loss: 0.0022 Acc: 0.9990 | Val Loss: 0.0258 Acc: 0.9901 | Prec: 0.9907 Rec: 0.9862 F1: 0.9885 ROC_AUC: 0.9996
Modelo salvo em resnet50_run4.pth
Histórico salvo em CSV em resnet50_run4_history.csv
```

# Possível abordagem


                                    ┌───────────────┐
                                    │  Input Image  │
                                    └───────┬───────┘
                                            │
          ┌──────────────────────────────────────────────────────────────────┐
          │                 │                  │                             │              
          ▼                 ▼                  ▼                             ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            ┌───────────────┐    
  │  Backbone     │  │  Backbone     │  │  Backbone     │            │   Handcraft   │        
  │  ResNet50     │  │  VGG16        │  │  MobileNet    │            │  extraction   │
  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘            └───────┬───────┘
          │                  │                  │                            │
          ▼                  ▼                  ▼                            │  
  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐                   │
  │ AvgPool +      │ │ AvgPool +      │ │ AvgPool +      │                   │  
  │ Flatten        │ │ Flatten        │ │ Flatten        │                   │  
  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘                   │  
          │                  │                  │                            │  
          ▼                  ▼                  ▼                            │  
  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐           ┌───────────────┐
  │ Interm. Layer  │ │ Interm. Layer  │ │ Interm. Layer  │           │ Feature Fusion│   
  │ 512 → 128      │ │ 1024 → 256     │ │ 256 → 64       │────────── │   Concatenate │
  │ BN + Dropout   │ │ BN + Dropout   │ │ BN + Dropout   │           └───────┬───────┘
  │ ReLU           │ │ ReLU           │ │ ReLU           │                   │ 
  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘                   │ 
          │                  │                  │                            │ 
          ▼                  ▼                  ▼                            ▼ 
  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐           ┌────────────────┐
  │ Classifier     │ │ Classifier     │ │ Classifier     │           │   Classifier   │
  │ (Softmax/FC)   │ │ (Softmax/FC)   │ │ (Softmax/FC)   │           │   (AdaBoost)   │ 
  └────────────────┘ └────────────────┘ └────────────────┘           └────────────────┘



