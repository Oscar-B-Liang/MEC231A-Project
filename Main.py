<<<<<<< HEAD
<<<<<<< HEAD
from kuka_bullet import KukaBullet
=======
# from kuka_bullet import KukaBullet
>>>>>>> Zedai-Yang
=======
from kuka_bullet import KukaBullet
>>>>>>> Zedai-Yang
import System_Parameters as Para
import MPC
import numpy as np
import Main_func as Func
import os
<<<<<<< HEAD
<<<<<<< HEAD
import pybullet as p
from matplotlib import pyplot as plt


def main():
=======
# import pybullet as p
=======
import pybullet as p
>>>>>>> Zedai-Yang
from matplotlib import pyplot as plt


def run_pybullet(x0, xf, horizon, ts):
>>>>>>> Zedai-Yang

    # Connect to physics engine and GUI.
    physicsClientId = p.connect(p.GUI)
    p.resetSimulation(physicsClientId=physicsClientId)
    p.setGravity(0, 0, -9.8)
    root_path = os.path.join(os.path.abspath(os.getcwd()), 'gyms')
    robot = KukaBullet(physicsClientId=physicsClientId, root_path=root_path)
    p.stepSimulation(physicsClientId=physicsClientId)

    # The end-effector should always be pointing down.
    quat_desire = np.asarray(robot.graspTargetQuat)

<<<<<<< HEAD
    eef_pos = np.zeros((2000, 3))
    ft_readings = np.zeros((2000))

    fig = plt.figure(figsize=(15, 10))
    axs = [plt.subplot(4, 1, i + 1) for i in range(4)]

    x0 = np.array([0.50, 0, 0, 0, 0])
    xf = np.array([0.65, 0, 0, 0, 4])
    Para.set_horizon(2000)
    Para.set_time_step(0.001)
    Para.set_initial_state(x0)
    Para.set_final_state(xf)
    Para.update_system_para()
    [model, feasibility, x_opt, u_opt, j_opt] = MPC.solve_mpc()
    print(feasibility)
    # print('xOpt =', x_opt)
    # print('uOpt = ', u_opt)
    # _input = Func.input_modify(u_opt)
=======
    eef_pos = np.zeros((horizon, 3))
    eef_vel = np.zeros((horizon, 3))
    ft_readings = np.zeros((horizon))

    fig = plt.figure(figsize=(15, 10))
    axs = [plt.subplot(5, 1, i + 1) for i in range(5)]

    feasibility, x_opt, u_opt = Func.run_openloop_mpc(x0, xf, horizon, ts)
    depth_mpc = Func.calculate_depth(x_opt[2, :], x_opt[3, :], x_opt[4, :])
    depth_desired = Para.get_depth_desired(x_opt[0, :], x_opt[1, :])
>>>>>>> Zedai-Yang

    robot.reset_robot()
    robot.step_robot(steps=1, sleep=False)
    robot.set_gains(Kp=[200, 200, 100, 100, 100, 100], Kd=[10, 10, 20, 20, 20, 20], Kqp=[50] * 7, Kqd=[5] * 7, Ko=[50] * 7, Kf=1.0)
    desired_height = robot.graspTargetPos[2]
    print(desired_height)

    # Stablizing the force torque sensor.
    for step in range(240 * 3 * 6):
        robot.step_robot(steps=1, sleep=False)
        target = np.asarray(robot.graspTargetPos)

        tau = robot.compute_torque(target, quat_desire, fz_desired=(step / 240 / 6), use_ext_tau=True, nullspace_type='full', restPoses=None)
        robot.apply_torque(tau)

<<<<<<< HEAD
    for step in range(2000):
=======
    for step in range(horizon):
>>>>>>> Zedai-Yang
        desired_height = robot.graspTargetPos[2]
        target = np.array([x_opt[0, step], x_opt[1, step], desired_height])
        desired_vel = np.array([x_opt[2, step], x_opt[3, step], 0.0, 0.0, 0.0, 0.0])
        tau = robot.compute_torque(target, quat_desire, vel_desire=desired_vel, fz_desired=x_opt[4, step], use_ext_tau=True, nullspace_type='full', restPoses=None)
        robot.apply_torque(tau)
        robot.step_robot(steps=1, sleep=False)
        eef_pos[step] = robot.x_pos.reshape(-1,)
<<<<<<< HEAD
        ft_readings[step] = robot.fz

=======
        eef_vel[step] = robot.x_quat.reshape(-1,)  # read velocity, need change
        ft_readings[step] = robot.fz

    depth_simulate = Func.calculate_depth(eef_vel[:, 0], eef_vel[:, 1], ft_readings)

>>>>>>> Zedai-Yang
    for j in range(3):
        axs[j].plot(eef_pos[:, j])
        if j == 0:
            axs[j].plot(x_opt[0, :])

    axs[0].set_ylim([0.45, 0.7])
    axs[1].set_ylim([-0.01, 0.01])
    axs[2].set_ylim([0.5, 0.7])

    axs[3].plot(ft_readings)
    axs[3].plot(x_opt[4, :])
    axs[3].set_ylim([-10, 10])

<<<<<<< HEAD
=======
    axs[4].plot(x_opt[0, :], x_opt[1, :] + 0.5 * depth_mpc, color='blue')
    axs[4].plot(x_opt[0, :], x_opt[1, :] - 0.5 * depth_mpc, color='blue')
    axs[4].fill_between(x_opt[0, :], x_opt[1, :] - 0.5 * depth_mpc, x_opt[1, :] + 0.5 * depth_mpc, facecolor='blue',
                        alpha=0.1)
    axs[4].plot(x_opt[0, :], x_opt[1, :] + 0.5 * depth_desired, color='red')
    axs[4].plot(x_opt[0, :], x_opt[1, :] - 0.5 * depth_desired, color='red')
    axs[4].fill_between(x_opt[0, :], x_opt[1, :] - 0.5 * depth_desired, x_opt[1, :] + 0.5 * depth_desired,
                        facecolor='red', alpha=0.1)
    axs[4].plot(eef_pos[:, 0], eef_pos[:, 1] + 0.5 * depth_desired, color='green')
    axs[4].plot(eef_pos[:, 0], eef_pos[:, 1] - 0.5 * depth_desired, color='green')
    axs[4].fill_between(eef_pos[:, 0], eef_pos[:, 1] - 0.5 * depth_desired, eef_pos[:, 1] + 0.5 * depth_desired,
                        facecolor='green', alpha=0.1)

>>>>>>> Zedai-Yang
    # axs[4].plot(ft_readings + ())
    plt.show()


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    run_pybullet(x0=[0.50, 0, 0, 0], xf=[0.65, 0, 0, 0], horizon=2000, ts=0.001)
    # horizon_min = Func.find_smallest_horizon(x0=[0.50, 0, 0, 0], xf=[0.65, 0, 0, 0], ts=0.001)
    # feasibility, x_opt, u_opt = Func.run_openloop_mpc(x0=[0.50, 0, 0, 0], xf=[0.65, 0, 0, 0], horizon=2000, ts=0.001)
>>>>>>> Zedai-Yang
