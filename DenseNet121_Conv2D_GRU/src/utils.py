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