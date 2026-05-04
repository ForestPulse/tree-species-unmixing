#!/usr/bin/env python
from matplotlib import cm
import tensorflow as tf
from keras.saving import register_keras_serializable
import numpy as np
from tqdm import tqdm
import random
import os
import argparse
import ast
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the pure data numpy array", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_c_noMapPap" )
parser.add_argument("--num_models", help="number of models you want to create", default= 10)
parser.add_argument("--year", help="year of synthetic mixture", default= '2021')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", 
                    #default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','OtherDT', 'Ground', 'Shadow']")
                    default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Birke','Erle','OtherDT', 'Ground', 'Shadow']")
parser.add_argument("--num_hidden_layer", help="number of hidden layer", default= 5) # orig 5
parser.add_argument("--hidden_layer_nodes", help="number of nodes per hidden layer", default = 128) # orig 128
parser.add_argument("--learning_rate", help="learning_rate for training", default = 1e-3)
parser.add_argument("--batch_size", help="the batch size for training", default = 256) # orig 256
parser.add_argument("--epochs", help="number of epochs", default = 125)

parser.add_argument("--local", help="check if model is calculated for a local tileset", default= 'FALSE')
parser.add_argument("--tile", help="tile-set of the local model application", default= 'X0055_Y0053')

args = parser.parse_args()

def train(model_number, pos, tile=args.tile):
    #------------------------------ added --------------------------------
    @register_keras_serializable(package="Custom")
    class SumToOneLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)
    #---------------------------------------------------------------------
    
    def norm(a):
        a_out = a/10000.
        return a_out
    
    def get_model(input_shape, lc_num, hidden_layer_num, hidden_layer_node):
        def dense(x, filter_size):
            layer = tf.keras.layers.Dense(filter_size)(x)
            return layer
        x_in = tf.keras.Input(shape=(input_shape,))
        #x = tf.keras.layers.Flatten()(x_in) # theroretisch redundant
        x = x_in 

        # architecture of hidden layers
        x = tf.keras.layers.ReLU()(dense(x, 128))
        x = tf.keras.layers.ReLU()(dense(x, 256))
        x = tf.keras.layers.ReLU()(dense(x, 512))
        x = tf.keras.layers.ReLU()(dense(x, 256))
        x = tf.keras.layers.ReLU()(dense(x, 128))

        # Version with sum to one condition before outputting the layer
        x_out = dense(x, lc_num)
        x_out = SumToOneLayer()(x_out)
        
        model =tf.keras.Model(inputs = x_in, outputs = x_out)
        #print(model.summary())
        return model
    
    def get_loss(x, y, model, training=True):
        y_pred = model(x, training=training)
        # Version with sum to one condition
        loss = tf.keras.losses.MeanAbsoluteError()(y, y_pred)
        # Version without sum to one condition
        #loss = tf.keras.losses.MeanSquaredError()(y, y_pred)
        return loss

    @tf.function
    def train(x, y, model, opt):
        with tf.GradientTape() as tape:
            loss = get_loss(x, y, model, training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    # load the synthetic mixed data
    x_mixed_out_path = os.path.join(args.working_directory, '2_mixed_data_glob','version' +str(model_number) , 'x_mixed_' + str(args.year) + '.npy')
    y_mixed_out_path = os.path.join(args.working_directory, '2_mixed_data_glob','version' +str(model_number) , 'y_mixed_' + str(args.year) + '.npy')
    x_train = np.load(x_mixed_out_path)
    y_train = np.load(y_mixed_out_path)

    x_train = norm(x_train)

    #----------
    #x_test = x_train[0:56000]
    #y_test = y_train[0:56000]
    #x_train = x_train[56000:]
    #y_train = y_train[56000:]
    #----------

    # define parameter
    input_shape = (x_train.shape[1])
    lc_num = len(  ast.literal_eval(args.tree_labels))
    hidden_layer_num = args.num_hidden_layer
    hidden_layer_node = args.hidden_layer_nodes
    
    model = get_model(input_shape, lc_num, hidden_layer_num, hidden_layer_node)

    lr = float(args.learning_rate)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    train_index = list(range(y_train.shape[0]))
    
    batch_size = int(args.batch_size)
    iterations = int(y_train.shape[0]/batch_size)
    epochs = int(args.epochs)
    random.shuffle(train_index)

    if not os.path.exists( os.path.join(args.working_directory, '3_trained_model_glob','version' +str(model_number))):
        os.makedirs( os.path.join(args.working_directory, '3_trained_model_glob' ,'version' +str(model_number)))

    with open(os.path.join(args.working_directory, '3_trained_model_glob' ,'version' +str(model_number),'performance.txt'), 'w') as file:
        file.write(f"Epoch;MAE;time\n")

    #------------------------- Training -----------------------------------
    start_time = time.time()
    pbar = tqdm(total=epochs, desc=f"Model {model_number}", position=pos, leave=True)
    for e in range(epochs):
        loss_train = 0
        for i in range(iterations):
            x_batch = x_train[train_index[i*batch_size: i*batch_size + batch_size]]
            y_batch = y_train[train_index[i*batch_size: i*batch_size + batch_size]]
            loss_train += train(x_batch, y_batch, model, opt)

        loss_train = loss_train / iterations
        loss_train = loss_train.numpy()
    
        #print('Epoch: ', e)
        #print('MAE: ', loss_train)
        passed_time_min = int((time.time() - start_time) // 60)
        passed_time_sec = (time.time() - start_time) - (passed_time_min * 60)
        with open(os.path.join(args.working_directory, '3_trained_model_glob','version' + str(model_number),'performance.txt'), 'a') as file:
            file.write(f"{e};{loss_train};{str(passed_time_min)}:{str(int(round(passed_time_sec,0))) }\n")
        random.shuffle(train_index)
        if e > 30: #vllt besser 100
            opt.learning_rate = lr/10
            if e > 75: # dann 140; 150 Ende
                opt.learning_rate = lr/100
                if e > 100: # 
                    opt.learning_rate = lr/1000

        pbar.update(1)
        pbar.set_description(f"Model {model_number} | Epoch {e+1} | MAE={loss_train:.4f}")
    pbar.close()

    # ----------- neue Version---------------
    # def lr_schedule(epoch, lr):
    #     if epoch < 30:
    #         return 1e-3
    #     elif epoch < 75:
    #         return 1e-4
    #     else:
    #         return 1e-5
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    # # Optimizer und Loss
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    #     #loss=masked_mae,
    #     loss='MeanAbsoluteError',
    #     #loss=tf.keras.losses.KLDivergence(), <- klappt nicht, wegen vieler 0en im Vecotr
    #     #metrics=[tf.keras.metrics.MeanAbsoluteError(),'accuracy', masked_mae]
    #     metrics=[tf.keras.metrics.MeanAbsoluteError(),'accuracy']
    # )
    # # Early Stopping Callback
    # early_stop = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',   # oder 'val_accuracy'
    #     #monitor='val_accuracy',   # oder 'val_loss'
    #     patience=15,              # stoppt, wenn keine Verbesserung für 15 Epochen
    #     restore_best_weights=True
    # )

    # # Training
    # history = model.fit(
    #     x_train, y_train,
    #     validation_data=(x_test, y_test),
    #     epochs=125,                # max. Epochs
    #     batch_size=64,
    #     callbacks=[early_stop, lr_scheduler],
    #     verbose=2
    # )

    # save the trained model
    model_path = os.path.join(args.working_directory, '3_trained_model_glob','version' + str(model_number), 'saved_model'+ str(model_number)+ '.keras')
    tf.keras.models.save_model(model, model_path)

    # -----------------------------
    # Lernkurven speichern
    # with open(os.path.join(args.working_directory, '3_trained_model_glob',
    #                        'version' + str(model_number), 'training_log.txt'), 'w') as f:
    #     #f.write("Epoch\tTrain_Loss\tVal_Accuracy\n")
    #     #for epoch, (tl, va) in enumerate(zip(history.history['loss'], history.history['val_accuracy']), start=1):
    #     #    f.write(f"{epoch}\t{tl:.6f}\t{va:.6f}\n")
    #     f.write("Epoch\tTrain_Loss\tTrain_MAE\tTrain_Acc\tVal_Loss\tVal_MAE\tVal_Acc\n")
    #     for i in range(len(history.history['loss'])):
    #         f.write(
    #             f"{i+1}\t"
    #             f"{history.history['loss'][i]:.6f}\t"
    #             f"{history.history['mean_absolute_error'][i]:.6f}\t"
    #             f"{history.history['accuracy'][i]:.6f}\t"
    #             f"{history.history['val_loss'][i]:.6f}\t"
    #             f"{history.history['val_mean_absolute_error'][i]:.6f}\t"
    #             f"{history.history['val_accuracy'][i]:.6f}\n"
    #     )

    return
    # -----------------------------

    # ----- plot the performance graph -----
    df = pd.read_csv(os.path.join(args.working_directory, '3_trained_model_glob' ,'version' +str(model_number),'performance.txt'), sep=';', engine='python')
    epochs = df['Epoch'].values
    mae_values = df['MAE'].values
    # --- Plot erstellen ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mae_values, marker='o', linestyle='-', markersize=4, color='tab:blue')

    plt.title("Training MAE über Epochen")
    plt.xlabel("Epoche")
    plt.ylabel("MAE")
    # Achsenbereiche
    plt.xlim(-5, int(args.epochs))
    plt.ylim(0, 0.12)
    # Gitterlinien bei 0.02, 0.04, 0.06, 0.08, 0.1
    plt.yticks(np.arange(0, 0.12, 0.02))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Layout & Speichern
    plt.tight_layout()
    plt.savefig(os.path.join(args.working_directory, '3_trained_model_glob', 'version' +str(model_number),'mae_plot.png'), dpi=300)
    plt.close()
    #print('Performance plot saved.')
                    
def plot(tile=args.tile):
    versions = os.listdir(os.path.join(args.working_directory, '3_trained_model_glob' ))
    files = [os.path.join(args.working_directory, '3_trained_model_glob', version , 'performance.txt') for version in sorted(versions) if version.startswith('version')]
    # --- Farben vorbereiten (10 Blautöne) ---
    cmap = plt.get_cmap('Blues', len(files) + 3)
    colors = [cmap(i + 3) for i in range(len(files))]  # überspringe die sehr hellen

    # --- Plot erstellen ---
    plt.figure(figsize=(9, 5))

    for i, (file, color) in enumerate(zip(files, colors), start=1):
        df = pd.read_csv(file, sep=';', engine='python', header=0)
        plt.plot(df['Epoch'], df['MAE'],
             label=f"Modell {i}",
             marker='o', linestyle='-', markersize=3,
             color=color,
             linewidth=1.8)
    plt.title("MAE-Verlauf aller Modelle")
    plt.xlabel("Epoche")
    plt.ylabel("MAE")
    # Achsenbereiche & Gitter
    plt.xlim(-5, int(args.epochs))
    plt.ylim(0, 0.12)
    plt.yticks(np.arange(0, 0.12, 0.02))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Legende rechts neben dem Plot
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Modelle")
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Platz für Legende rechts

    # --- Speichern ---
    plt.savefig(os.path.join(args.working_directory, '3_trained_model_glob', 'mae_plot.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    #for i in range(int(args.num_models)):
    #    train(i+50,1)
    #train(5,1)
    num_workers = 10
    ####Parallel(n_jobs=num_workers, backend="loky")(delayed(train)(i+1, (i%num_workers)*2) for i in range(int(args.num_models)))
    Parallel(n_jobs=num_workers, backend="loky")(delayed(train)(i+1, i) for i in range(int(args.num_models)))
    plot()

    #list_tile_sets = ['Y05_X01','Y05_X02','Y05_X03']
    #for tile_set in list_tile_sets:
    #    Parallel(n_jobs=num_workers, backend="loky")(delayed(train)(i+1, i,tile_set) for i in range(int(args.num_models)))
    #    plot(tile_set)