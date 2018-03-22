import numpy as np
from numpy import dot
from numpy.linalg import inv
from util.clip import clipState, clipAngle
import math


class EIFModel:
    def __init__(self, dimension, robotFeaturesDim, envFeaturesDim, motionModel, mesModel, covMes, muInitial):
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.dimension = dimension

        self.HH = np.eye(dimension)
        self.H = np.eye(dimension)
        self.b = dot(muInitial.T, self.H)
        self.bb = dot(muInitial.T, self.H)
        self.S = np.zeros(dimension * robotFeaturesDim).reshape((dimension, robotFeaturesDim))
        self.S[:robotFeaturesDim] = np.eye(robotFeaturesDim)
        self.invZ = inv(covMes)
        self.motionModel = motionModel
        self.mesModel = mesModel

    def update(self, measures, landmarkIds, command, U):
        self.__motion_update(command, U)
        for ldmIndex, ldmMes in zip(landmarkIds, measures):
            self.__measurement_update(ldmMes, int(ldmIndex))
        return self.H, self.b

    def __motion_update(self, command, U):
        previousMeanState = self.estimate()
        meanStateChange = self.motionModel.exact_move(previousMeanState, command)
        newMeanState = clipState(previousMeanState + meanStateChange)

        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.H)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        IA = np.eye(self.H.shape[0]) + gradMeanMotion  # TO IMPROVE
        sigma = dot(dot(IA, inv(self.H)), IA.T) + dot(dot(self.S, U), self.S.T)
        self.H = inv(sigma)
        self.b = dot((newMeanState).T, self.H)
        self.HH = self.H.copy()
        self.bb = self.b.copy()

    def __measurement_update(self, ldmMes, ldmIndex):
        mu = self.estimate()
        meanMes, gradMeanMes = self.__get_mean_measurement_params(mu, ldmIndex)

        z = np.array(ldmMes).reshape(len(ldmMes), 1)
        zM = np.array(meanMes).reshape(len(ldmMes), 1)
        C = gradMeanMes

        mesError = (z - zM)
        mesError[1, 0] = clipAngle(mesError[1, 0], force=True)
        mesError += dot(C.T, mu)
        mesError[1, 0] = clipAngle(mesError[1, 0])
        self.H += dot(dot(C, self.invZ), C.T)
        self.b += dot(dot(mesError.T, self.invZ), C.T)

    def __get_mean_measurement_params(self, mu, ldmIndex):
        realIndex = self.robotFeaturesDim + ldmIndex * self.envFeaturesDim
        ldmMeanState = mu[realIndex: realIndex + self.envFeaturesDim]
        rMeanState = mu[:self.robotFeaturesDim]

        meanMes = self.mesModel.measureFunction(rMeanState, ldmMeanState)
        gradMeanMes = self.mesModel.gradMeasureFunction(rMeanState, ldmMeanState, realIndex)
        return meanMes, gradMeanMes

    def estimate(self, H=None, b=None):
        H = self.H if H is None else H
        b = self.b if b is None else b
        return clipState(dot(b, inv(H)).T)
