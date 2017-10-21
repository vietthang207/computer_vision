from utils import *

"""Implement two background subtraction algorithms. Using by creating an instance of
class and sequentially apply each frame in a video. apply() method returns a bitmap of foreground
for the current frame based on history of applied frames.
Background subtraction using Average is fast but poor in extracting foreground objects
Back ground subtraction using Mixtures of Gaussian is much more accurate in determining foreground objects
but run very slow. Need to optimize further."""

DEFAULT_NOISE_SIGMA = 30.0 * 0.5
DEFAULT_VAR = DEFAULT_NOISE_SIGMA * DEFAULT_NOISE_SIGMA * 4.0


class BackgroundSubtractorAvg:

    def __init__(self, threshold=10):
        self.nframes = 0
        self.threshold = threshold
        self.background = np.zeros((0, 0, 0))
        self.nchannels = 1

    def apply(self, frame):
        if self.nframes == 0:
            self.background = frame
            self.nchannels = frame.shape[2]
        else:
            self.background = np.multiply(self.background, self.nframes/(self.nframes+1.0)) + np.multiply(frame, 1/(self.nframes + 1.0))

        diff = frame - self.background
        sqr_diff = np.square(diff)
        bitmap = np.zeros((self.background.shape[0], self.background.shape[1]))
        bitmap[np.sum(sqr_diff.astype(np.uint8), axis=self.nchannels-1) > self.threshold] = 255
        self.nframes += 1
        return bitmap

    def get_background(self):
        return self.background.astype(np.uint8)
        # return self.background


class BackgroundSubtractorMOG:

    nframes = 0

    def __init__(self, nmixtures=3, backgroundRatio=0.7, varThreshold=2.5, learningRate=0.01):
        PixelMixture.nmixtures = nmixtures
        PixelMixture.backgroundRato = backgroundRatio
        PixelMixture.varThreshold = varThreshold * varThreshold
        PixelMixture.learningRate = learningRate
        BackgroundSubtractorMOG.nframes = 0
        self.domain = np.empty((0, 0), dtype=object)

    def apply(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]
        nchannels = frame.shape[2]
        bitmap = np.zeros((height, width))
        if BackgroundSubtractorMOG.nframes == 0:
            PixelMixture.nchannels = nchannels
            self.domain = np.empty((height, width), dtype=object)
            for i in range(height):
                for j in range(width):
                    newPixel = PixelMixture()
                    newPixel.calculate(frame[i][j])
                    self.domain[i][j] = newPixel
        else:
            for i in range(height):
                for j in range(width):
                    bitmap[i][j] = self.domain[i][j].calculate(frame[i][j])

        BackgroundSubtractorMOG.nframes += 1
        return bitmap

    def release(self):
        BackgroundSubtractorMOG.nframes = 0


class PixelMixture:

    nmixtures = 3
    backgroundRatio = 0.7
    varThreshold = 2.5 * 2.5
    nchannels = 3
    learningRate = 0.01
    
    def __init__(self):
        self.weight = np.zeros((PixelMixture.nmixtures, 1))
        self.mean = np.zeros((PixelMixture.nmixtures, PixelMixture.nchannels))
        #assume same variance for all channels
        self.var = np.zeros((PixelMixture.nmixtures, 1))

    def calculate(self, pixel):
        kHit = -1
        kForeground = -1
        if self.weight[0] == 0:
            self.weight[0] = 1
            self.mean[0] = pixel
            for i in range(PixelMixture.nmixtures):
                self.var[0] = DEFAULT_NOISE_SIGMA * DEFAULT_NOISE_SIGMA * 4.0
            kHit = 0
            kForeground = 1
        else:
            if PixelMixture.learningRate > 0:
                for k in range(PixelMixture.nmixtures):
                    diff = pixel - self.mean[k]
                    if sqr(diff) < PixelMixture.varThreshold * PixelMixture.nchannels * self.var[k]:
                        #update new metrics
                        new_weight = self.weight[k] - PixelMixture.learningRate * (1.0 - self.weight[k])
                        r = PixelMixture.learningRate * gauss(pixel, self.mean[k], self.var[k])
                        new_mean = self.mean[k] + r * diff
                        new_var = self.var[k] + r * (np.dot(diff, diff) - self.var[k])
                        self.weight[k] = new_weight
                        self.mean[k] = new_mean
                        self.var[k] = max(new_var, DEFAULT_VAR/4)

                        #sort
                        prev_sortKey = self.weight[k] / self.var[k]
                        prev_weight = self.weight[k]
                        prev_mean = self.mean[k]
                        prev_var = self.var[k]
                        k1 = k - 1
                        if k1 >= 0:
                            next_sortKey = self.weight[k1] / self.var[k1]
                            next_weight = self.weight[k1]
                            next_mean = self.mean[k1]
                            next_var = self.var[k1]
                            while k1 >= 0 and next_sortKey < prev_sortKey:
                                self.weight[k1] = prev_weight
                                self.mean[k1] = prev_mean
                                self.var[k1] = prev_var
                                self.weight[k1+1] = next_weight
                                self.mean[k1+1] = next_mean
                                self.var[k1+1] = next_var

                                prev_sortKey = next_sortKey
                                prev_weight = next_weight
                                prev_mean = next_mean
                                prev_var = next_var
                                next_sortKey = self.weight[k1-1] / self.var[k1-1] if k1 > 0 else 0.0
                                next_weight = self.weight[k1-1] if k1 > 0 else 0.0
                                next_mean = self.mean[k1-1] if k1 > 0 else np.zeros(len(prev_mean))
                                next_var = self.var[k1-1] if k1 > 0 else 0.0
                                k1 -= 1

                        kHit = k1 + 1
                        break

                if kHit < 0:
                    #remove the weakest mixture
                    kHit = k = PixelMixture.nmixtures - 1
                    self.weight[k] = 0.05
                    self.mean[k] = pixel
                    self.var[k] = DEFAULT_NOISE_SIGMA * DEFAULT_NOISE_SIGMA * 4.0

                wsum = 0.0
                for k in range(PixelMixture.nmixtures):
                    wsum += self.weight[k]

                wscale = 1/wsum
                wsum = 0.0
                for k in range(PixelMixture.nmixtures):
                    self.weight[k] *= wscale
                    wsum += self.weight[k]
                    if wsum > PixelMixture.backgroundRatio and kForeground < 0:
                        kForeground = k + 1
            else:
                for k in range(PixelMixture.nmixtures):
                    diff = pixel - self.mean[k]
                    if sqr(diff) < PixelMixture.varThreshold * PixelMixture.nchannels * self.var[k]:
                        kHit = k
                        break

                if kHit >= 0:
                    wsum = 0
                    for k in range(PixelMixture.nmixtures):
                        wsum += self.weight[k]
                        if wsum > PixelMixture.backgroundRatio:
                            kForeground = k + 1
                            break

        if kHit < 0 or kHit >= kForeground:
            return 255
        return 0
