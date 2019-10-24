from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Activation, GRU, TimeDistributed
from keras.optimizers import Adam

class VideoClassifier:
    
    def __init__(self, input_shape, optimizer=Adam()):
        self.densenet_shape = input_shape[1:]
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.model = None
    
    def build_model(self):
        densenet_inputs = Input(shape=self.densenet_shape)
        densenet = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_tensor=densenet_inputs
        )
        densenet_x = densenet.layers[-1].output
        densenet_x = GlobalAveragePooling2D()(densenet_x)
        densenet_model = Model(inputs=densenet.inputs, outputs=densenet_x)

        sequence_input = Input(shape=self.input_shape)
        x = TimeDistributed(densenet_model)(sequence_input)
        x = GRU(256)(x)
        x = Dropout(0.2)(x)
        x = Dense(2)(x)
        predictions = Activation('softmax')(x)
        self.model = Model(inputs=sequence_input, outputs=predictions)
        self.model.compile(self.optimizer, loss='categorical_crossentropy')