import brainpy as bp
import brainpy.math as bm
import numpy as np
import os

class HD_cell(bp.DynamicalSystem):
    def __init__(
        self,
        num,
        noise_stre=0.01,
        tau=1.0,
        tau_v=10.0,
        k=1.0,
        mbar=15.,
        a=0.4,
        A=3.0,
        J0=4.0,
        z_min=-bm.pi,
        z_max=bm.pi,
    ):
        super(HD_cell, self).__init__()

        # parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        self.k = k / num * 20  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0 / num * 20  # maximum connection value
        self.m = mbar * tau / tau_v
        self.noise_stre = noise_stre

        # neuron num
        self.num = num  # head-direction cell
        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        x1 = bm.linspace(z_min, z_max, num + 1)  # The encoded feature values
        self.x = x1[0:-1]
        self.rho = num / self.z_range  # The neural density
        self.dx = self.z_range / num  # The stimulus density

        # connection matrix
        conn_mat = self.make_conn()
        self.conn_fft = bm.fft.fft(conn_mat)

        # neuron state variables
        self.r = bm.Variable(bm.zeros(num))
        self.u = bm.Variable(bm.zeros(num))  # head direction cell
        self.v = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))
        self.center = bm.Variable(bm.zeros(1))
        self.centerI = bm.Variable(bm.zeros(1))

        # 定义积分器
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def dist(self, d):
        d = self.circle_period(d)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self):
        d = self.dist(bm.abs(self.x[0] - self.x))
        Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (2 * bm.pi * self.a ** 2)
        return Jxx

    def get_center(self, r, x):
        exppos = bm.exp(1j * x)
        center = bm.angle(bm.sum(exppos * r))
        return center.reshape(-1,)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def circle_period(self, A):
        B = bm.where(A > bm.pi, A - 2 * bm.pi, A)
        B = bm.where(B < -bm.pi, B + 2 * bm.pi, B)
        return B

    def input_HD(self, HD):
        # integrate self motion
        return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - HD) / self.a))

    def reset_state(self, HD_truth):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.center.value = bm.Variable(bm.zeros(1,) + HD_truth)

    def update(self, HD, ThetaInput):
        self.center = self.get_center(r=self.r, x=self.x)
        Iext = ThetaInput * self.input_HD(HD)
        # Calculate input
        r_fft = bm.fft.fft(self.r)
        Irec = bm.real(bm.fft.ifft(r_fft * self.conn_fft))
        input_total = Iext + Irec + bm.random.randn(self.num) * self.noise_stre
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share.load("t"), input_total, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2


class PC_cell(bp.DynamicalSystem):
    def __init__(
        self,
        num,
        noise_stre=0.01,
        tau=1.0,
        tau_v=10.0,
        k=1.0,
        mbar=15.,
        a=0.4,
        A=3.0,
        J0=4.0,
        z_min=-bm.pi,
        z_max=bm.pi,
    ):
        super(PC_cell, self).__init__()

        # parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        self.k = k / num * 20  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0 / num * 20  # maximum connection value
        self.m = mbar * tau / tau_v
        self.noise_stre = noise_stre

        # neuron num
        self.num = num  # head-direction cell
        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        x1 = bm.linspace(z_min, z_max, num + 1)  # The encoded feature values
        self.x = x1[0:-1]
        self.rho = num / self.z_range  # The neural density
        self.dx = self.z_range / num  # The stimulus density

        # connection matrix
        conn_mat = self.make_conn()
        self.conn_fft = bm.fft.fft(conn_mat)

        # neuron state variables
        self.r = bm.Variable(bm.zeros(num))
        self.u = bm.Variable(bm.zeros(num))  # head direction cell
        self.v = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))
        self.center = bm.Variable(bm.zeros(1))
        self.centerI = bm.Variable(bm.zeros(1))

        # 定义积分器
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def dist(self, d):
        d = self.circle_period(d)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self):
        d = self.dist(bm.abs(self.x[0] - self.x))
        Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (2 * bm.pi * self.a ** 2)
        return Jxx

    def get_center(self, r, x):
        exppos = bm.exp(1j * x)
        center = bm.angle(bm.sum(exppos * r))
        return center.reshape(-1,)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def circle_period(self, A):
        B = bm.where(A > bm.pi, A - 2 * bm.pi, A)
        B = bm.where(B < -bm.pi, B + 2 * bm.pi, B)
        return B

    def input_HD(self, HD):
        # integrate self motion
        return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - HD) / self.a))

    def reset_state(self, HD_truth):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.center.value = bm.Variable(bm.zeros(1,) + HD_truth)

    def update(self, HD, ThetaInput):
        self.center = self.get_center(r=self.r, x=self.x)
        Iext = ThetaInput * self.input_HD(HD)
        # Calculate input
        r_fft = bm.fft.fft(self.r)
        Irec = bm.real(bm.fft.ifft(r_fft * self.conn_fft))
        input_total = Iext + Irec + bm.random.randn(self.num) * self.noise_stre
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share.load("t"), input_total, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2


def calculate_inst_speed(directions, samples_per_sec):
    diff_dist = np.diff(directions.flatten())
    # consider the periodic boundary condition that is, if diff > pi, then diff = diff - 2*pi
    # if diff < -pi, then diff = diff + 2*pi
    diff_dist = np.where(diff_dist > np.pi, diff_dist - 2 * np.pi, diff_dist)
    diff_dist = np.where(diff_dist < -np.pi, diff_dist + 2 * np.pi, diff_dist)
    inst_speed = diff_dist * samples_per_sec
    # insert the first element the same as the second element
    inst_speed = np.insert(inst_speed, 0, 0)
    return inst_speed


def create_directory_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def circle_period(d):
    d = np.where(d > np.pi, d - 2 * np.pi, d)
    d = np.where(d < -np.pi, d + 2 * np.pi, d)
    return d


def traj(x0, v, T):
    x = []
    xt = x0
    for i in range(T):
        xt = xt + v * bm.dt
        if xt > np.pi:
            xt -= 2 * np.pi
        if xt < -np.pi:
            xt += 2 * np.pi
        x.append(xt)
    return np.array(x)

def calculate_inst_speed(directions, samples_per_sec):
    diff_dist = np.diff(directions.flatten())
    # consider the periodic boundary condition that is, if diff > pi, then diff = diff - 2*pi
    # if diff < -pi, then diff = diff + 2*pi
    diff_dist = np.where(diff_dist > np.pi, diff_dist - 2 * np.pi, diff_dist)
    diff_dist = np.where(diff_dist < -np.pi, diff_dist + 2 * np.pi, diff_dist)
    inst_speed = diff_dist * samples_per_sec
    # insert the first element the same as the second element
    inst_speed = np.insert(inst_speed, 0, 0)
    return inst_speed


def create_directory_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def period(data):
    # 计算傅里叶变换
    fft_x = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), d=(bm.dt))

    # 仅使用正频率部分
    positive_frequencies = frequencies[np.where(frequencies >= 0)]
    positive_fft_x = np.abs(fft_x[np.where(frequencies >= 0)])

    # 找到最大频率分量
    dominant_frequency = positive_frequencies[np.argmax(positive_fft_x)]
    dominant_period = 1 / dominant_frequency

    # 打印结果
    print(f"Dominant Frequency: {dominant_frequency} Hz")
    print(f"Dominant Period: {dominant_period} ")


def circle_period(d):
    d = np.where(d > np.pi, d - 2 * np.pi, d)
    d = np.where(d < -np.pi, d + 2 * np.pi, d)
    return d


def traj(x0, v, T):
    x = []
    xt = x0
    for i in range(T):
        xt = xt + v * bm.dt
        if xt > np.pi:
            xt -= 2 * np.pi
        if xt < -np.pi:
            xt += 2 * np.pi
        x.append(xt)
    return np.array(x)
