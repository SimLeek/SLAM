import numpy as np
from util.clip import clipState, clipAngle
import math
from numpy import dot
from numpy.linalg import inv


class EKFModel:
    def __init__(self, dimension, robotFeaturesDim, envFeaturesDim, motionModel, mesModel, covMes, muInitial):
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.dimension = dimension

        self.Sigma = np.eye(dimension)
        self.mu = muInitial.copy()
        self.S = np.zeros(dimension * robotFeaturesDim).reshape((dimension, robotFeaturesDim))
        self.S[:robotFeaturesDim] = np.eye(robotFeaturesDim)
        self.Z = covMes
        self.motionModel = motionModel
        self.mesModel = mesModel

    def update(self, measures, landmarkIds, command, U):
        self.__motion_update(command, U)
        for ldmIndex, ldmMes in zip(landmarkIds, measures):
            self.__measurement_update(ldmMes, int(ldmIndex))
        return self.Sigma, self.mu

    def __motion_update(self, command, U):
        previousMeanState = self.mu
        meanStateChange = self.motionModel.exact_move(previousMeanState, command)
        newMeanState = clipState(previousMeanState + meanStateChange)

        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.Sigma)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        IA = np.eye(self.Sigma.shape[0]) + gradMeanMotion  # TO IMPROVE
        self.mu = newMeanState
        self.Sigma = dot(dot(IA, self.Sigma), IA.T) + dot(dot(self.S, U), self.S.T)

    def __measurement_update(self, ldmMes, ldmIndex):
        mu = self.mu
        Sigma = self.Sigma
        meanMes, gradMeanMes = self.__get_mean_measurement_params(mu, ldmIndex)

        z = np.array(ldmMes).reshape(len(ldmMes), 1)
        zM = np.array(meanMes).reshape(len(ldmMes), 1)
        C = gradMeanMes

        toInvert = inv(dot(dot(C.T, Sigma), C) + self.Z)
        K = dot(dot(Sigma, C), toInvert)

        mesError = (z - zM)
        mesError[1, 0] = clipAngle(mesError[1, 0], force=True)
        mesError = dot(K, mesError)
        mesError[1, 0] = clipAngle(mesError[1, 0])

        self.mu += mesError
        self.Sigma = dot(np.eye(self.dimension) - dot(K, C.T), Sigma)

    def __get_mean_measurement_params(self, mu, ldmIndex):
        realIndex = self.robotFeaturesDim + ldmIndex * self.envFeaturesDim
        ldmMeanState = mu[realIndex: realIndex + self.envFeaturesDim]
        rMeanState = mu[:self.robotFeaturesDim]

        meanMes = self.mesModel.measureFunction(rMeanState, ldmMeanState)
        gradMeanMes = self.mesModel.gradMeasureFunction(rMeanState, ldmMeanState, realIndex)
        return meanMes, gradMeanMes

    def estimate(self):
        return self.mu
