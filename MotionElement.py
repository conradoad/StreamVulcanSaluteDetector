import numpy as np
import cv2 as cv

class MotionElement:

    image: np.ndarray
    start_frame_ts: float
    has_transparency: bool
    ttl_sec: float
    finished: bool
    
    # postition and dimension
    pos_normalized: (float, float)
    size_normalized: (float, float)
    
    # motion settings
    translate_rate: (float, float)
    resize_rate: float
    
    def __init__(self, 
                 image_path: np.ndarray,
                 height: int,
                 has_transparency: bool = False,
                 pos_normalized = (0.0, 0.0),
                 size_normalized = (1.0, 1.0),
                 translate_rate = (0.0, 0.0),
                 resize_rate = 0.0,
                 ttl_sec: float = 5) -> None:
        self.image = cv.imread(image_path, cv.IMREAD_UNCHANGED if has_transparency else None)
        width = int(self.image.shape[1] * height / self.image.shape[0])
        self.image = cv.resize(src=self.image, dsize=(height, width), interpolation=cv.INTER_CUBIC)
        
        self.start_frame_ts = None
        self.has_transparency = has_transparency
        self.pos_normalized = pos_normalized
        self.size_normalized = size_normalized
        self.translate_rate = translate_rate
        self.resize_rate = resize_rate
        self.ttl_sec = ttl_sec
        self.finished = False
        
    def annotate(self, frame: np.ndarray, frame_timestamp_ms: float) -> np.ndarray:
        
        if self.start_frame_ts == None:
            self.start_frame_ts = frame_timestamp_ms
            
        seconds_elapsed = ((frame_timestamp_ms - self.start_frame_ts) / 1000)
        
        if seconds_elapsed > self.ttl_sec:
            self.finished = True
            return frame
        
        pos_normalized = (self.pos_normalized[0] + self.translate_rate[0] * seconds_elapsed,
                          self.pos_normalized[1] + self.translate_rate[1] * seconds_elapsed)
                
        if self.resize_rate != 0:
            new_h = int(self.image.shape[0] *(1 + seconds_elapsed * self.resize_rate))
            new_w = int(self.image.shape[1] * (1 + seconds_elapsed * self.resize_rate))
            
            if new_h <= 0 or new_w <= 0: return frame
            
            image = cv.resize(src=self.image, dsize=(new_h, new_w), interpolation=cv.INTER_CUBIC)
        else:
            image = self.image
                
        
        if pos_normalized[0] > 1 or pos_normalized[1] > 1: return frame
        
        frame_y1 = int(frame.shape[0] * pos_normalized[0])
        frame_x1 = int(frame.shape[1] * pos_normalized[1])
        
        if frame_y1 < 0:
            image = image[-frame_y1:image.shape[0],:]
            frame_y1 = 0
        
        if frame_x1 < 0:
            image = image[:,-frame_x1:image.shape[1]]
            frame_x1 = 0
                    
        frame_y2 = frame_y1 + image.shape[0]        
        frame_x2 = frame_x1 + image.shape[1]
        
        if frame_y2 < 0 or frame_x2 < 0: return frame
        
        if frame_y2 > frame.shape[0]:
            image = image[:(image.shape[0] - (frame_y2 - frame.shape[0])),:]
            frame_y2 = frame.shape[0]
            
        if frame_x2 > frame.shape[1]:
            image = image[:,:(image.shape[1] - (frame_x2 - frame.shape[1]))]
            frame_x2 = frame.shape[1]
        
        
        if self.has_transparency:
            condition = np.stack((image[:,:,3],) * 3, axis=-1) == 0
            frame[frame_y1:frame_y2, frame_x1:frame_x2] = np.where(condition, frame[frame_y1:frame_y2, frame_x1:frame_x2], image[:,:,0:3])
        else:
            frame[frame_y1:frame_y2, frame_x1:frame_x2] = image
        
        return frame
    