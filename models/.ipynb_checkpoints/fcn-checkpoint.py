from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, PReLU
from tensorflow.keras import regularizers

def build_model(in_dim, args):
    print(args.num_neuron)
    model = Sequential()
    model.add(Dense(128, kernel_initializer='normal', input_dim = in_dim))
    model.add(PReLU())
    model.add(Dense(64, kernel_initializer='normal'))
    model.add(PReLU())
    model.add(Dense(32, kernel_initializer='normal'))
    model.add(PReLU())
    model.add(Dense(args.num_neuron, kernel_initializer='normal'))
    model.add(PReLU())
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    return model

def build_model_L2_regular(in_dim, args):
    print(args.num_neuron)
    model = Sequential()
    model.add(Dense(128, 
                    kernel_initializer='normal', 
                    kernel_regularizer=regularizers.l2(l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    input_dim = in_dim))
    model.add(PReLU())
    model.add(Dense(64, kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),))
    model.add(PReLU())
    model.add(Dense(32, kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),))
    model.add(PReLU())
    model.add(Dense(args.num_neuron, kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),))
    model.add(PReLU())
    model.add(Dense(1, kernel_initializer='normal', 
                    kernel_regularizer=regularizers.l2(l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activation='linear'))
    return model