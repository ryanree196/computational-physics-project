
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Computational Physics: Projectile Motion
# Numerical integration using the RK4 method.

G = 9.81
INITIAL_SPEED = 30.0
LAUNCH_ANGLE = 45.0
DRAG = 0.15
DT = 0.01

OUTPUT = Path("results")
OUTPUT.mkdir(exist_ok=True)


def derivatives(state):
    x, y, vx, vy = state
    ax = -DRAG * vx
    ay = -G - DRAG * vy
    return np.array([vx, vy, ax, ay])


def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + dt * k1 / 2)
    k3 = derivatives(state + dt * k2 / 2)
    k4 = derivatives(state + dt * k3)

    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def analytical_solution(t):
    angle = np.radians(LAUNCH_ANGLE)
    vx0 = INITIAL_SPEED * np.cos(angle)
    vy0 = INITIAL_SPEED * np.sin(angle)

    decay = np.exp(-DRAG * t)

    x = vx0 * (1 - decay) / DRAG
    y = (
        (vy0 + G / DRAG) * (1 - decay) / DRAG
        - G * t / DRAG
    )

    return x, y


def simulate():
    angle = np.radians(LAUNCH_ANGLE)

    state = np.array([
        0.0,
        0.0,
        INITIAL_SPEED * np.cos(angle),
        INITIAL_SPEED * np.sin(angle)
    ])

    states = [state.copy()]
    times = [0.0]

    while True:
        next_state = rk4_step(state, DT)

        if next_state[1] < 0:
            fraction = state[1] / (state[1] - next_state[1])
            impact = state + fraction * (next_state - state)
            states.append(impact)
            times.append(times[-1] + fraction * DT)
            break

        state = next_state
        states.append(state.copy())
        times.append(times[-1] + DT)

    return np.array(times), np.array(states)


def main():
    times, states = simulate()

    x = states[:, 0]
    y = states[:, 1]

    exact_x, exact_y = analytical_solution(times)

    error = np.sqrt(
        (x - exact_x)**2 + (y - exact_y)**2
    )

    print("PROJECTILE MOTION SIMULATION")
    print("-" * 40)
    print(f"Maximum height: {np.max(y):.4f} m")
    print(f"Horizontal range: {x[-1]:.4f} m")
    print(f"Flight duration: {times[-1]:.4f} s")
    print(f"Maximum position error: {np.max(error):.8f} m")
    print(f"Mean position error: {np.mean(error):.8f} m")

    np.savetxt(
        OUTPUT / "simulation_results.csv",
        np.column_stack((times, x, y, exact_x, exact_y, error)),
        delimiter=",",
        header="time,x,y,analytical_x,analytical_y,error",
        comments=""
    )

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, label="RK4 numerical solution")
    plt.plot(exact_x, exact_y, "--", label="Analytical solution")
    plt.xlabel("Horizontal Distance (m)")
    plt.ylabel("Height (m)")
    plt.title("Projectile Motion with Linear Air Resistance")
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT / "trajectory_comparison.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(times, error)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Position error (m)")
    plt.title("Numerical Accuracy of RK4")
    plt.grid()
    plt.savefig(OUTPUT / "numerical_error.png")
    plt.close()

    print("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()
