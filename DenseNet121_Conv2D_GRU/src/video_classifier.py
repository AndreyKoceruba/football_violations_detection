import plotly.graph_objects as go
import plotly.offline as pyo
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Activation, GRU, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class VideoClassifier:
    
    def __init__(self, input_shape, optimizer=Adam()):
        self.densenet_shape = input_shape[1:]
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.model = None
        self.history = None
    
    def set_model(self, model):
        self.model = model
    
    def build_model(self):
        densenet_inputs = Input(shape=self.densenet_shape)
        densenet = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_tensor=densenet_inputs
        )
        densenet.trainable = False
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
        
    def fit(
        self,
        generator,
        epochs=1,
        steps_per_epoch=None,
        verbose=2,
        validation_data=None,
        validation_steps=None,
        use_multiprocessing=False,
        class_weight=None,
        checkpoint_path='model/checkpoint_best_model.h5'
    ):
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(patience=4, verbose=1)
        reduce_lr = ReduceLROnPlateau(patience=2, factor=0.5)
        callbacks = [checkpoint, early_stopping, reduce_lr]
        self.history = self.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            workers=1,
            use_multiprocessing=use_multiprocessing,
            class_weight=class_weight
        )
        return self.history
    
    def plot_learning_curves(self, filename='plot/learning_curves.html'):
        train_loss = go.Scatter(
            x=list(range(1, len(self.history.history['loss']) + 1)),
            y=history.history['loss'],
            mode='lines+markers',
            name='Train loss',
            hoverinfo='y'
        )
        val_loss = go.Scatter(
            x=list(range(1, len(self.history.history['val_loss']) + 1)),
            y=history.history['val_loss'],
            mode='lines+markers',
            name='Validation loss',
            hoverinfo='y'
        )

        data = [train_loss, val_loss]
        layout = go.Layout(
            title=dict(
                text='Learning curves'
            )
        )
        fig = go.Figure(data=data, layout=layout)
        pyo.plot(fig, filename=filename)