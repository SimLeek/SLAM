from scipy import sparse
from scipy.sparse import linalg
import numpy as np
from numpy import dot
from numpy.linalg import inv
from util.clip import clipState, clipAngle
import math
from util.dots import dots


class SEIFModel:

    def __init__(self, dimension,
                 robotFeaturesDim,
                 envFeaturesDim,
                 motionModel,
                 mesModel,
                 covMes,
                 muInitial,
                 maxLinks):
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.dimension = dimension

        self.H = np.eye(dimension)
        self.b = dot(muInitial.T, self.H)
        self.mu = muInitial.copy()

        self.Sx = np.zeros(dimension * robotFeaturesDim).reshape((dimension, robotFeaturesDim))
        self.Sx[:robotFeaturesDim] = np.eye(robotFeaturesDim)
        self.invZ = inv(covMes)
        self.motionModel = motionModel
        self.mesModel = mesModel
        self.maxLinks = maxLinks

    def update(self, measures, landmarkIds, command, U):
        self._motion_update_sparse(command, U)
        self._mean_update()
        for ldmIndex, ldmMes in zip(landmarkIds, measures):
            self._measurement_update(ldmMes, int(ldmIndex))
        self._mean_update()
        self._sparsification()
        return self.H, self.b, self.mu

    def _motion_update(self, command, U):
        r = self.robotFeaturesDim
        previousMeanState = self.estimate()
        meanStateChange = self.motionModel.exact_move(previousMeanState, command)
        newMeanState = clipState(previousMeanState + meanStateChange)

        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.H)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        delta = dots(self.Sx.T, gradMeanMotion, self.Sx)
        G = dots(self.Sx, (inv(np.eye(r) + delta) - np.eye(r)), self.Sx.T)
        phi = np.eye(self.dimension) + G
        Hp = dots(phi.T, self.H, phi)
        deltaH = dots(Hp, self.Sx, inv(inv(U) + dots(self.Sx.T, Hp, self.Sx)), self.Sx.T, Hp)
        H = Hp - deltaH
        self.H = H
        self.b = dot(newMeanState.T, self.H)
        self.mu = newMeanState

    def _motion_update_sparse(self, command, U):
        r = self.robotFeaturesDim
        previousMeanState = self.estimate()
        meanStateChange, _ = self.motionModel.exact_move(previousMeanState, command)
        newMeanState = clipState(previousMeanState + meanStateChange)

        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.H)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        Sx = sparse.bsr_matrix(self.Sx)
        sH = sparse.bsr_matrix(self.H)
        invU = sparse.coo_matrix(inv(U))
        sparseGradMeanMotion = sparse.bsr_matrix(gradMeanMotion)

        delta = Sx.T.dot(sparseGradMeanMotion).dot(Sx)
        try:
            G = Sx.dot(linalg.inv(sparse.eye(r) + delta) - sparse.eye(r)).dot(Sx.T)
        except RuntimeError: # happens on singular matrix. We can't be _too_ perfect.
            G = Sx.dot(Sx.T)

        phi = sparse.eye(self.dimension) + G
        Hp = phi.T.dot(sH).dot(phi)
        deltaH = Hp.dot(Sx).dot(linalg.inv(invU + Sx.T.dot(Hp).dot(Sx))).dot(Sx.T).dot(Hp)
        # H = inv(Hp + dots(self.Sx, U, self.Sx.T))
        H = Hp - deltaH
        # self.b = self.b - dot(previousMeanState.T, self.H - H) + dot(meanStateChange.T, H)
        self.H = H.todense()
        self.b = H.dot(newMeanState).T
        self.mu = newMeanState

    def _mean_update(self):
        ''' Coordinate ascent '''
        mu = self.mu
        iterations = 30
        y0, yp = self._partition_links()
        y = np.concatenate([np.arange(self.robotFeaturesDim), y0, yp])

        # vMu = dot(self.b, inv(self.H)).T
        # muSave = []
        # muSave2 = []

        for t in range(iterations):
            for i in y:
                y2 = np.setdiff1d(y, i)
                mu[i] = (self.b[0, i] - dot(self.H[i, y2], mu[y2])) / self.H[i, i]
            # muSave.extend([np.linalg.norm(mu - vMu)])
        self.mu = mu
        # plt.plot(muSave)

    def _measurement_update(self, ldmMes, ldmIndex):
        mu = self.mu
        meanMes, gradMeanMes = self._get_mean_measurement_params(mu, ldmIndex)

        z = np.array(ldmMes).reshape(len(ldmMes), 1)
        zM = np.array(meanMes).reshape(len(ldmMes), 1)
        C = gradMeanMes

        mesError = (z - zM)
        mesError[1, 0] = clipAngle(mesError[1, 0], force=True)
        correction = mesError + dot(C.T, mu)
        correction[1, 0] = clipAngle(correction[1, 0])
        self.H += dot(dot(C, self.invZ), C.T)
        self.b += dot(dot(correction.T, self.invZ), C.T)

    def _partition_links(self):
        r = self.robotFeaturesDim
        e = self.envFeaturesDim
        d = self.dimension
        l = (d - r) / e
        arrRF = np.arange(r)

        norms = np.array([np.linalg.norm(self.H[arrRF][:, np.arange(i * e + r, (i + 1) * e + r)]) for i in range(int(l))])
        ids = np.argsort(norms)
        yp = ids[-self.maxLinks:]
        y0 = np.setdiff1d(np.where(norms > 0), yp)

        yp = np.concatenate([np.arange(y * e, (y + 1) * e) for y in yp]) + r
        if len(y0) > 0:
            y0 = np.concatenate([np.arange(y * e, (y + 1) * e) for y in y0]) + r

        return y0, yp

    def _build_projection_matrix(self, indices):
        d1 = self.H.shape[0]
        d2 = len(indices)

        S = np.zeros((d1, d2))
        S[indices] = np.eye(d2)
        return S

    def _sparsification(self):
        x = np.arange(self.robotFeaturesDim)
        y0, yp = self._partition_links()
        Sx = sparse.coo_matrix(self._build_projection_matrix(x))
        Sy0 = sparse.coo_matrix(self._build_projection_matrix(y0))
        Sxy0 = sparse.coo_matrix(self._build_projection_matrix(np.concatenate((x, y0))))
        Sxyp = sparse.coo_matrix(self._build_projection_matrix(np.concatenate((x, yp))))
        Sxy0yp = sparse.coo_matrix(self._build_projection_matrix(np.concatenate((x, y0, yp))))
        H = sparse.bsr_matrix(self.H)

        Hp = Sxy0yp.dot(Sxy0yp.T).dot(H).dot(Sxy0yp).dot(Sxy0yp.T)

        Ht = H - (0 if not y0.size else Hp.dot(Sy0).dot(linalg.inv(Sy0.T.dot(Hp).dot(Sy0))).dot(Sy0.T).dot(Hp)) \
             + Hp.dot(Sxy0).dot(linalg.inv(Sxy0.T.dot(Hp).dot(Sxy0))).dot(Sxy0.T).dot(Hp) \
             - H.dot(Sx).dot(linalg.inv(Sx.T.dot(H).dot(Sx))).dot(Sx.T).dot(H)
        eps = 1e-5
        Htt = Ht.todense()
        Htt[np.abs(Htt) < eps] = 0
        bt = self.b + (Ht - H).dot(self.mu)

        self.H = Htt
        self.b = bt

    def _get_mean_measurement_params(self, mu, ldmIndex):
        realIndex = self.robotFeaturesDim + ldmIndex * self.envFeaturesDim
        ldmMeanState = mu[realIndex: realIndex + self.envFeaturesDim]
        rMeanState = mu[:self.robotFeaturesDim]

        meanMes = self.mesModel.measureFunction(rMeanState, ldmMeanState)
        gradMeanMes = self.mesModel.gradMeasureFunction(rMeanState, ldmMeanState, realIndex)
        return meanMes, gradMeanMes

    def estimate(self):
        return self.mu
