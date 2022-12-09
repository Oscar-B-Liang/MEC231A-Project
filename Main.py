from kuka_bullet import KukaBullet
import System_Parameters as Para
import MPC
import numpy as np
import Main_func as Func
import os
import pybullet as p
from matplotlib import pyplot as plt


def run_pybullet(x0, xf, horizon, ts):

    # Connect to physics engine and GUI.
    physicsClientId = p.connect(p.GUI)
    p.resetSimulation(physicsClientId=physicsClientId)
    p.setGravity(0, 0, -9.8)
    root_path = os.path.join(os.path.abspath(os.getcwd()), 'gyms')
    robot = KukaBullet(physicsClientId=physicsClientId, root_path=root_path)
    p.stepSimulation(physicsClientId=physicsClientId)

    # The end-effector should always be pointing down.
    quat_desire = np.asarray(robot.graspTargetQuat)

    eef_pos = np.zeros((horizon, 3))
    eef_vel = np.zeros((horizon, 3))
    ft_readings = np.zeros((horizon))

    feasibility, x_opt, u_opt = Func.run_openloop_mpc(x0, xf, horizon, ts)
    depth_mpc = Func.calculate_depth(x_opt[2, :], x_opt[3, :], x_opt[4, :])
    depth_desired = Para.get_depth_desired(x_opt[0, :], x_opt[1, :])

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

    for step in range(horizon):
        desired_height = robot.graspTargetPos[2]
        target = np.array([x_opt[0, step], x_opt[1, step], desired_height])
        desired_vel = np.array([x_opt[2, step], x_opt[3, step], 0.0, 0.0, 0.0, 0.0])
        tau = robot.compute_torque(target, quat_desire, vel_desire=desired_vel, fz_desired=x_opt[4, step], use_ext_tau=True, nullspace_type='full', restPoses=None)
        robot.apply_torque(tau)
        robot.step_robot(steps=1, sleep=False)
        eef_pos[step] = robot.x_pos.reshape(-1,)
        eef_vel[step] = robot.dx_linear.reshape(-1,)  # read velocity, need change
        ft_readings[step] = robot.fz

    depth_simulate = Func.calculate_depth(eef_vel[:, 0], eef_vel[:, 1], ft_readings)

    fig_1 = plt.figure(figsize=(5, 5))
    plt.plot(x_opt[0, :], label="computed")
    plt.plot(eef_pos[:, 0], label="actual")
    plt.ylim([0.45, 0.7])
    plt.title("X Coordinate")
    plt.legend()
    plt.savefig("figures/x_coordinate.png")

    fig_2 = plt.figure(figsize=(5, 5))
    plt.plot(x_opt[4, :], label="computed")
    plt.plot(ft_readings, label="actual")
    plt.ylim([-5, 5])
    plt.title("Pressing Force")
    plt.legend()
    plt.savefig("figures/pressing_force.png")

    fig_3 = plt.figure(figsize=(5, 5))
    plt.plot(x_opt[0, :], x_opt[1, :] + 0.5 * depth_mpc, color='blue', label="computed")
    plt.plot(x_opt[0, :], x_opt[1, :] - 0.5 * depth_mpc, color='blue')
    plt.fill_between(x_opt[0, :], x_opt[1, :] - 0.5 * depth_mpc, x_opt[1, :] + 0.5 * depth_mpc, facecolor='blue', alpha=0.1)
    plt.plot(x_opt[0, :], x_opt[1, :] + 0.5 * depth_desired, color='red', label="target")
    plt.plot(x_opt[0, :], x_opt[1, :] - 0.5 * depth_desired, color='red')
    plt.fill_between(x_opt[0, :], x_opt[1, :] - 0.5 * depth_desired, x_opt[1, :] + 0.5 * depth_desired, facecolor='red', alpha=0.1)
    plt.plot(eef_pos[:, 0], eef_pos[:, 1] + 0.5 * depth_desired[1:], color='green', label="actual")
    plt.plot(eef_pos[:, 0], eef_pos[:, 1] - 0.5 * depth_desired[1:], color='green')
    plt.fill_between(eef_pos[:, 0], eef_pos[:, 1] - 0.5 * depth_desired[1:], eef_pos[:, 1] + 0.5 * depth_desired[1:], facecolor='green', alpha=0.1)
    plt.title("Line Width")
    plt.legend()
    plt.savefig("figures/line_width.png")


if __name__ == "__main__":
    run_pybullet(x0=[0.50, 0, 0, 0], xf=[0.65, 0, 0, 0], horizon=2000, ts=0.001)
    # horizon_min = Func.find_smallest_horizon(x0=[0.50, 0, 0, 0], xf=[0.65, 0, 0, 0], ts=0.001)
    # feasibility, x_opt, u_opt = Func.run_openloop_mpc(x0=[0.50, 0, 0, 0], xf=[0.65, 0, 0, 0], horizon=2000, ts=0.001)
