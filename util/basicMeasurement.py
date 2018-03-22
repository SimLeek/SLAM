from .clip import clipAngle
import numpy as np

class BasicMeasurement:
    def __init__(self, covariance, robotFeaturesDim, envFeaturesDim, measureFunction, gradMeasureFunction, detectionSize=0, detectionCone=0):
        self.covariance = np.atleast_2d(covariance)
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.measureFunction = measureFunction
        self.gradMeasureFunction = gradMeasureFunction
        self.detectionSize = detectionSize
        self.detectionCone = detectionCone

    #  Input the real state
    def measure(self, state, noise=False):
        dim = state.shape[0]
        dimR = self.robotFeaturesDim
        dimE = self.envFeaturesDim
        rState = state[:dimR]
        envState = state[dimR:]
        nbLandmark = (dim - dimR) / dimE

        mes = np.zeros(int(nbLandmark * dimE)).reshape(int(nbLandmark), int(dimE)) # array to hold all landmarks
        landmarkIds = np.zeros(int(nbLandmark)) # array to hold all landmark ids
        j = 0

        for i, landmark in enumerate(envState.reshape((int(nbLandmark), int(dimE), 1))):
            diffNorm, diffAngle = self.measureFunction(rState, landmark)
            angleOk = (abs(clipAngle(diffAngle, True)) < self.detectionCone / 2.) or (self.detectionCone is 0)
            distanceOk = (diffNorm < self.detectionSize) or (self.detectionSize is 0)

            if distanceOk and angleOk:
                mes[j] = [diffNorm, diffAngle]
                landmarkIds[j] = i
                j += 1

        mes = mes[:j]
        landmarkIds = landmarkIds[:j]
        mes = np.array(mes)
        if noise:
            mes += self._get_noise(mes)
        return mes, landmarkIds

    def _get_noise(self, mes):
        noise = np.random.multivariate_normal(np.zeros(self.covariance.shape[0]), self.covariance, mes.shape[0])
        return noise