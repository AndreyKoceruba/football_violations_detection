import plotly.graph_objects as go
import plotly.offline as pyo

def scale_image(img):
    scaled_img = img / 255
    return scaled_img

def read_ids(filename):
    ids = []
    classes = []
    with open(filename) as f:
        next(f)
        for line in f:
            line = line.strip()
            id_, class_ = line.split(',')
            ids.append(id_)
            classes.append(class_)
    return ids, classes

def gini_score(roc_auc):
    return 2 * roc_auc - 1

def plot_learning_curves(history, output='plot/learning_curves.html'):
    
    train_loss = go.Scatter(
        x=list(range(1, len(history.history['loss']) + 1)),
        y=history.history['loss'],
        mode='lines+markers',
        name='Train loss',
        hoverinfo='y'
    )

    val_loss = go.Scatter(
        x=list(range(1, len(history.history['val_loss']) + 1)),
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
    pyo.plot(fig, filename=output)
   
def plot_curves(
    x_train,
    y_train,
    threshold_train,
    x_valid,
    y_valid,
    threshold_valid,
    x_test,
    y_test,
    threshold_test,
    curve_type='ROC',
    width=777,
    height=777,
    output='plot/curves.html'
):
    
    trace_train = go.Scatter(
        x=x_train,
        y=y_train,
        mode='lines',
        name='Train',
        text=threshold_train
    )
    
    trace_valid = go.Scatter(
        x=x_valid,
        y=y_valid,
        mode='lines',
        name='Valid',
        text=threshold_valid
    )
    
    trace_test = go.Scatter(
        x=x_test,
        y=y_test,
        mode='lines',
        name='Test',
        text=threshold_test
    )

    data = [trace_train, trace_valid, trace_test]
    
    if curve_type == 'ROC':
        
        x_title = 'FPR'
        y_title = 'TPR'
        title = 'ROC Curves'
        
        trace_dot = go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            showlegend=False,
            line=dict(
                dash='dot'
            )
        )
        
        data.append(trace_dot)
        
    elif curve_type == 'PR':
        
        x_title = 'Recall'
        y_title = 'Precision'
        title = 'PR Curves'
    
    layout = go.Layout(
        title=dict(
            text=title
        ),
        hovermode='closest',
        width=width,
        height=height,
        xaxis=dict(
            title=dict(
                text=x_title
            )
        ),
        yaxis=dict(
            title=dict(
                text=y_title
            )
        )
    )
    fig = go.Figure(data=data, layout=layout)
    pyo.plot(fig, filename=output)

def benchmark(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        delta = end_time - start_time
        minutes = int(delta / 60)
        seconds = int(delta - minutes * 60)
        print('Time of "{}": {} min. {} sec.'.format(func.__name__, minutes, seconds))
        return res
    return wrapper