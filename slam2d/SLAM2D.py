import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import math
from util.measureFunctions import measureFunction, gradMeasureFunction
from util.basicMeasurement import BasicMeasurement
from util.basicMovement import BasicMovement
from slam2d.seifModel import SEIFModel

import util.measureFunctions as meas

if __name__ == '__main__':

    dimension = None

    def simu():
        global dimension
        T = 100  # Number of timesteps
        nbLandmark = 900
        maxSpeed = 5
        maxRotation = 45 * math.pi / 180  # 45  # en radians
        sizeMap = 50

        # Robot Detection Parameters
        detectionSize = 2  # 40
        detectionCone = 180 * math.pi / 180  # en radians

        # Dimension Constants
        robotFeaturesDim = 3
        envFeaturesDim = 2
        commandsDim = 2
        mesDim = 2
        dimension = robotFeaturesDim + nbLandmark * envFeaturesDim
        meas.dimension = dimension

        # Covariances for motions and measurements
        covarianceMotion = np.eye(robotFeaturesDim)
        covarianceMotion[0, 0] = 1 ** 2  # motion noise variance X
        covarianceMotion[1, 1] = 1 ** 2  # motion noise variance Y
        covarianceMotion[2, 2] = (5 * math.pi / 180) ** 2  # motion noise variance Angle

        covarianceMeasurements = np.eye(mesDim)
        covarianceMeasurements[0, 0] = 1 ** 2  # measurement noise variance distance
        covarianceMeasurements[1, 1] = (5 * math.pi / 180) ** 2  # motion noise variance Angle


        ## ----------------------------------------------------------------------
        ## Simulation initialization

        ## -------------------
        ## State Definition

        # Real robot state
        state = np.zeros((dimension, 1))

        x = np.linspace(-sizeMap, sizeMap, np.sqrt(nbLandmark))
        y = np.linspace(-sizeMap, sizeMap, np.sqrt(nbLandmark))
        xv, yv = np.meshgrid(x, y)
        state[robotFeaturesDim:, 0] = np.vstack([xv.ravel(), yv.ravel()]).ravel(order="F")
        # state[robotFeaturesDim:] = np.random.rand(nbLandmark * envFeaturesDim).reshape(nbLandmark * envFeaturesDim, 1) * 300 - 150


        # Basic and EIF estimator for robot state
        mu = state.copy()
        mu[robotFeaturesDim:] += np.random.normal(0, covarianceMeasurements[0, 0], nbLandmark * envFeaturesDim).reshape(nbLandmark * envFeaturesDim, 1)
        muSEIF = mu.copy()

        ## --------------------
        ## Models Definition

        motionModel = BasicMovement(maxSpeed, maxRotation, covarianceMotion, measureFunction)
        measurementModel = BasicMeasurement(covarianceMeasurements, robotFeaturesDim, envFeaturesDim, measureFunction, gradMeasureFunction, detectionSize, detectionCone)
        seif = SEIFModel(dimension, robotFeaturesDim, envFeaturesDim, motionModel, measurementModel, covarianceMeasurements, mu, 4)

        mus_simple = np.zeros((T, dimension))
        mus_seif = np.zeros((T, dimension))
        states = np.zeros((T, dimension))

        mus_simple[0] = np.squeeze(mu)
        states[0] = np.squeeze(state)


        for t in range(1, T):
            print("\nIteration %d" % t)
            state, motionCommand = motionModel.noisy_move(state)
            measures, landmarkIds = measurementModel.measure(state)

            mu += motionModel.noisy_move(mu, motionCommand)[0]

            H, _, _ = seif.update(measures, landmarkIds, motionCommand, covarianceMotion)
            print(H, ' / ', H.size)

            muSEIF = seif.estimate()

            print(muSEIF[:3])


            mus_simple[t] = np.squeeze(mu)
            mus_seif[t] = np.squeeze(muSEIF)
            states[t] = np.squeeze(state)


        landmarks = state[robotFeaturesDim:].reshape(nbLandmark, 2)
        plt.figure()
        ax = plt.gca()
        for x, y in landmarks:
            ax.add_artist(Circle(xy=(x, y),
                          radius=detectionSize,
                          alpha=0.3))
        plt.scatter(landmarks[:, 0], landmarks[:, 1])

        plt.plot(states[:, 0], states[:, 1], color='m')
        plt.plot(mus_simple[:, 0], mus_simple[:, 1], color='y')
        plt.plot(mus_seif[:, 0], mus_seif[:, 1], color='k')

        plt.legend(['Real position', 'Simple estimate', 'SEIF estimate'])
        plt.title("{0} landmarks".format(nbLandmark))
        plt.show()

    import cProfile
    cProfile.run('simu()')
