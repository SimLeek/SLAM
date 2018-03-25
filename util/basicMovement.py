import numpy as np
import math
from .clip import clipState

class BasicMovement:
    def __init__(self, maxSpeed, maxRotation, covariance, measureFunction):
        self.maxSpeed = maxSpeed
        self.maxRotation = maxRotation
        self.measureFunction = measureFunction
        self.covariance = np.atleast_2d(covariance)

    #  Input the real state
    def noisy_move(self, state, covariance=None, command=None):
        noise = self._get_noise(covariance)
        idealMove, command = self.exact_move(state, command)
        realMove = self._noisy_move(state, idealMove, noise)
        newState = state + realMove
        return clipState(newState), command

    def __choose_command(self, state):
        speed = self.maxSpeed * np.random.rand()
        if (np.linalg.norm(state[:2]) > 100):
            _, rotation = self.measureFunction(state[:3], [[0], [0]])
            rotation = np.clip(rotation, -self.maxRotation, self.maxRotation)
        else:
            rotation = (np.random.rand() * 2 - 1) * self.maxRotation
        return [speed, rotation]

    def exact_move(self, state, command=None):
        command = self.__choose_command(state) if command is None else command
        # state: x,y,rot
        # command: velocity, rotation
        speed, rotation = command
        angle = state[2]
        deltaX = speed * math.cos(angle)
        deltaY = speed * math.sin(angle)

        move = np.zeros_like(state)
        move[:3, 0] = [deltaX, deltaY, rotation]
        return move, command

    def _noisy_move(self, state, idealMove, noise):
        noisyMove = idealMove[:3] + noise
        noisySpeed, _ = self.measureFunction(noisyMove[:3], np.zeros_like(noise)[:2])
        noisyRotation = noisyMove[2]

        maxs = [self.maxSpeed, self.maxRotation]
        mins = [0, -self.maxRotation]
        correctedCommand = np.clip([noisySpeed, noisyRotation], mins, maxs)
        return self.exact_move(state, correctedCommand)

    def _noisy_move2(self, state, idealMove, noise):
        noisyMove = np.zeros_like(state)
        noisyMove[:3] = idealMove[:3] + noise
        return noisyMove

    def _get_noise(self, covariance=None):
        covariance = self.covariance if covariance is None else covariance
        noise = np.random.multivariate_normal(np.zeros(covariance.shape[0]), covariance, 1).T
        return noise