import os
import cv2
import numpy as np
from .video_stream import VideoStream
from keras.utils import Sequence, to_categorical

class ImageSequenceGenerator(Sequence):
    
    def __init__(
        self,
        videos,
        ids,
        classes=None,
        class_names=None,
        augmentator=None,
        target_size=(224, 224),
        fps=8,
        sequence_time=3,
        shift_time=1,
        batch_size=2,
        shuffle=True,
        seed=None,
        fit_eval=True
    ):
        self.videos = videos
        self.ids = ids
        self.classes = classes
        self.class_names = class_names
        self.augmentator = augmentator
        self.target_size = target_size
        self.fps = fps
        self.sequence_time = sequence_time
        self.shift_time = shift_time
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.fit_eval = fit_eval
        
        self.timesteps = self.fps * self.sequence_time
        self.shift_frames = self.fps * self.shift_time
        
        self.skip_id = []
        self.start_positions = []
        self.indexes = None
        self.length = None
        
        self.__check_videos_total_time()
        self.__make_start_positions()
        self.on_epoch_end()

    def __check_videos_total_time(self):
        all_ids = os.listdir(self.videos)
        self.skip_id = []
        for id_ in all_ids:
            video_path = os.path.join(self.videos, id_)
            with VideoStream(video_path) as cap:
                frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_time = frames_cnt / fps
                if total_time < self.sequence_time:
                    self.skip_id.append(id_)
                    warn_message = 'Video {} has time {} less than sequence_time {} and will be skipped'.format(
                        id_,
                        total_time,
                        self.sequence_time
                    )
                    warnings.warn(ShortVideoWarning(warn_message))
    
    def __next_frame_step(self, fps):
        if self.fps is None:
            next_frame_step = 1
        else:
            next_frame_step = int(np.ceil(fps / self.fps))
        return next_frame_step
    
    def __make_start_positions(self):
        for id_ in self.ids:
            if id_ not in self.skip_id:
                video_path = os.path.join(self.videos, id_)
                with VideoStream(video_path) as cap:
                    frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    next_frame_step = self.__next_frame_step(fps)
                    start_positions = range(
                        0,
                        frames_cnt - self.timesteps * next_frame_step,
                        self.shift_frames * next_frame_step
                    )
                    for start_position in start_positions:
                        self.start_positions.append((id_, start_position))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.start_positions))
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        self.length = int(np.ceil(len(self.start_positions) / float(self.batch_size)))
        return self.length
    
    def __get_x(self, batch_start_positions):
        batch_x = []
        for id_, start_position in batch_start_positions:
            video_path = os.path.join(self.videos, id_)
            with VideoStream(video_path) as cap:
                fps = cap.get(cv2.CAP_PROP_FPS)
                next_frame_step = self.__next_frame_step(fps)
                frame_sequence = []
                current_pos = start_position
                for i in range(self.timesteps):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, self.target_size)
                    frame_sequence.append(frame)
                    current_pos += next_frame_step
                frame_sequence = np.array(frame_sequence)
                if self.augmentator is not None:
                    frame_sequence = self.augmentator.augment(frame_sequence)
            batch_x.append(frame_sequence)
        batch_x = np.array(batch_x)
        return batch_x
    
    def __get_y(self, batch_start_positions):
        batch_y = []
        for id_, start_position in batch_start_positions:
            index = self.ids.index(id_)
            batch_y.append(self.classes[index])
        batch_y = np.array(batch_y)
        batch_y = to_categorical(batch_y, num_classes=len(self.class_names))
        return batch_y
    
    def getitem(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_start_positions = [self.start_positions[i] for i in indexes]
        batch_x = self.__get_x(batch_start_positions)
        if self.fit_eval:
            batch_y = self.__get_y(batch_start_positions)
            return batch_x, batch_y
        return batch_x
    
    def __getitem__(self, idx):
        return self.getitem(idx)
        