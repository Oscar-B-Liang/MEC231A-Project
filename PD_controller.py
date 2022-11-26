import System_Parameters as Para


class PIDController:
    kp, kd = 0.0, 0.0

    D_tol = 0.0
    err_prev, err_crt = 0.0, 0.0

    target = 0.0
    output = 0.0

    def __init__(self):
        self.kp = 0.0
        self.kd = 0.0

    def set_coefficient(self, _kp, _kd):
        self.kp = _kp
        self.kd = _kd

    def set_target(self, _target):
        self.target = _target

    def set_d2tolerance(self, _d_tol):
        self.D_tol = _d_tol

    def get_output(self):
        return self.output

    def update(self, _input):
        self.err_crt = self.target - _input
        p = self.kp * self.err_crt
        err_dev = self.err_crt - self.err_prev
        self.err_prev = self.err_crt
        d = self.kd * err_dev
        self.output = p + d
