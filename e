Namespace(WL=1000, ds='UCR', i=0, n=5, path='/home/eyokano/datsets/UCR_Anomaly_FullData/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt')
Using device:cuda
0.2
Training data: (28000, 1)
Validation data: (7000, 1)
Testing data: (44795, 1)
28000
********************parameters********************
val_size = 0.2
batch_size = 32
stride = 1
window_size = 5000
z_dim = 10
hidden_dim = 50
rnn_hidden_dim = 50
num_layers = 1
bidirectional = True
cell = lstm
lr = 0.0003
if_scheduler = True
scheduler_step_size = 5
scheduler_gamma = 0.5
epoch = 10
early_stop = True
early_stop_tol = 10
weighted_loss = True
strategy = quadratic
dis_ar_iter = 1
adv_rate = 0.005
data_prefix = UCR_Anomaly_FullData/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt
best_model_path = best_models/UCR_Anomaly_FullData/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt/5000
result_path = results/UCR_Anomaly_FullData/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt
device = cuda
********************parameters********************
RNNAutoEncoder(
  (encoder): RNNEncoder(
    (linear1): Linear(in_features=1, out_features=50, bias=True)
    (linear2): Linear(in_features=100, out_features=10, bias=True)
    (rnn): LSTM(50, 50, bidirectional=True)
  )
  (decoder): RNNDecoder(
    (linear1): Linear(in_features=10, out_features=50, bias=True)
    (linear2): Linear(in_features=100, out_features=1, bias=True)
    (rnn): LSTM(50, 50, bidirectional=True)
  )
)
MLPDiscriminator(
  (dis): Sequential(
    (0): Linear(in_features=1, out_features=50, bias=True)
    (1): Tanh()
    (2): Linear(in_features=50, out_features=50, bias=True)
    (3): Tanh()
    (4): Linear(in_features=50, out_features=1, bias=True)
    (5): Sigmoid()
  )
)
********************Start training********************
[Epoch 1/10] current training loss is 0.00011024257401004434, val loss is 0.0012854202752976929, adv loss is 0.6930893063545227, time per epoch is 17.000189065933228
[Epoch 2/10] current training loss is 3.413471131352708e-05, val loss is 0.00015294402151533565, adv loss is 0.6955201625823975, time per epoch is 16.89928126335144
[Epoch 3/10] current training loss is 2.1677771655959077e-05, val loss is 0.00010913198401192175, adv loss is 0.6958852410316467, time per epoch is 16.83085060119629
[Epoch 4/10] current training loss is 1.617832822375931e-05, val loss is 8.341762033448593e-05, adv loss is 0.6948179602622986, time per epoch is 16.81464982032776
[Epoch 5/10] current training loss is 1.161938598670531e-05, val loss is 6.591548776603478e-05, adv loss is 0.6948314309120178, time per epoch is 16.79576086997986
[Epoch 6/10] current training loss is 9.474164471612312e-06, val loss is 5.6939224940267555e-05, adv loss is 0.6944506168365479, time per epoch is 16.79725933074951
[Epoch 7/10] current training loss is 8.034153324842919e-06, val loss is 4.924813050326125e-05, adv loss is 0.6940812468528748, time per epoch is 16.794700622558594
[Epoch 8/10] current training loss is 6.824283900641603e-06, val loss is 4.212683489175492e-05, adv loss is 0.6944395303726196, time per epoch is 16.84015154838562
[Epoch 9/10] current training loss is 5.816671091452008e-06, val loss is 3.546436687568649e-05, adv loss is 0.6940463185310364, time per epoch is 16.812748432159424
[Epoch 10/10] current training loss is 4.866086328547681e-06, val loss is 2.9722152116244864e-05, adv loss is 0.6938115358352661, time per epoch is 16.82105016708374
F1 score is [0.03058 / 0.72887] (before adj / after adj), auc score is 0.47678.
Precision score is [0.01553 / 0.57341], recall score is [1.00000 / 1.00000].
{'f1': 0.030576070901033977, 'auc': 0.4767816240859145, 'precision': 0.015525388134703368, 'recall': 1.0, 'accuracy': 0.01555, 'dataset': 'UCR_Anomaly_FullData/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt', 'WL': 1000, 'n': 5, 'id': 0}
