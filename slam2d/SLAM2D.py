import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import math

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
        muEKF = mu.copy()
        muEIF = mu.copy()
        muSEIF = mu.copy()

        ## --------------------
        ## Models Definition

        motionModel = BasicMovement(maxSpeed, maxRotation, covarianceMotion, measureFunction)
        measurementModel = BasicMeasurement(covarianceMeasurements, robotFeaturesDim, envFeaturesDim, measureFunction, gradMeasureFunction, detectionSize, detectionCone)
        ekf = EKFModel(dimension, robotFeaturesDim, envFeaturesDim, motionModel, measurementModel, covarianceMeasurements, mu)
        eif = EIFModel(dimension, robotFeaturesDim, envFeaturesDim, motionModel, measurementModel, covarianceMeasurements, mu)
        seif = SEIFModel(dimension, robotFeaturesDim, envFeaturesDim, motionModel, measurementModel, covarianceMeasurements, mu, 4)

        mus_simple = np.zeros((T, dimension))
        mus_ekf = np.zeros((T, dimension))
        mus_eif = np.zeros((T, dimension))
        mus_seif = np.zeros((T, dimension))
        states = np.zeros((T, dimension))

        mus_simple[0] = np.squeeze(mu)
        mus_ekf[0] = np.squeeze(muEKF)
        mus_eif[0] = np.squeeze(muEIF)
        mus_seif[0] = np.squeeze(muEIF)
        states[0] = np.squeeze(state)


        # LOG Initial state
        print("BEFORE")
        print("EIF estimate :")
        print(muEIF)
        print("EKF estimate :")
        print(muEKF)
        print("Real state :")
        print(state)
        print('\n')

        for t in range(1, T):
            print("\nIteration %d" % t)
            state, motionCommand = motionModel.noisy_move(state)
            measures, landmarkIds = measurementModel.measure(state)

            mu += motionModel.exact_move(mu, motionCommand)

            H, _ = ekf.update(measures, landmarkIds, motionCommand, covarianceMotion)
            print (H != 0).sum(), ' / ', H.size
            H, _= eif.update(measures, landmarkIds, motionCommand, covarianceMotion)
            print (H != 0).sum(), ' / ', H.size
            H, _, _ = seif.update(measures, landmarkIds, motionCommand, covarianceMotion)
            print (H != 0).sum(), ' / ', H.size

            muEKF = ekf.estimate()
            muEIF = eif.estimate()
            muSEIF = seif.estimate()

            print "np.linalg.norm(muEIF-muSEIF)"
            print np.linalg.norm(muEIF-muSEIF)
            print np.linalg.norm(eif.b - seif.b)
            print np.linalg.norm(eif.H - seif.H)
            print muEIF[:3]
            print muSEIF[:3]


            mus_simple[t] = np.squeeze(mu)
            mus_ekf[t] = np.squeeze(muEKF)
            mus_eif[t] = np.squeeze(muEIF)
            mus_seif[t] = np.squeeze(muSEIF)
            states[t] = np.squeeze(state)


        # LOG Final state
        print('\n')
        print('AFTER')
        print("EIF estimate :")
        print(muEIF)
        print("EKF estimate :")
        print(muEKF)
        print("Real state :")
        print(state)
        print("Final Error EIF:")
        print(state - muEIF)
        print("Final Error EKF:")
        print(state - muEKF)
        print("Final Max Error EIF: %f" % max(state-muEIF))
        print("Final Norm Error EIF: %f" % np.linalg.norm(state-muEIF))
        print("Final Max Error EKF: %f" % max(state-muEKF))
        print("Final Norm Error EKF: %f" % np.linalg.norm(state-muEKF))
        print("Final Max Error SEIF: %f" % max(state-muSEIF))
        print("Final Norm Error SEIF: %f" % np.linalg.norm(state-muSEIF))

        landmarks = state[robotFeaturesDim:].reshape(nbLandmark, 2)
        plt.figure()
        ax = plt.gca()
        for x, y in landmarks:
            ax.add_artist(Circle(xy=(x, y),
                          radius=detectionSize,
                          alpha=0.3))
        plt.scatter(landmarks[:, 0], landmarks[:, 1])

        plt.plot(states[:, 0], states[:, 1])
        plt.plot(mus_simple[:, 0], mus_simple[:, 1])
        plt.plot(mus_ekf[:, 0], mus_ekf[:, 1])
        plt.plot(mus_eif[:, 0], mus_eif[:, 1])
        plt.plot(mus_seif[:, 0], mus_seif[:, 1])

        plt.legend(['Real position', 'Simple estimate', 'EKF estimate', 'EIF estimate', 'SEIF estimate'])
        plt.title("{0} landmarks".format(nbLandmark))
        plt.show()

    import cProfile
    cProfile.run('simu()')
