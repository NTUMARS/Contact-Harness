# franka_controllers

ROS 1 controller package for the Franka robot. It is adapted from an open-source Franka controller package, reduced to two controllers. The robot server in `harness/franka/` needs it.

## Content

| File | Origin |
|---|---|
| `src/cartesian_impedance_controller.cpp`, `include/franka_controllers/cartesian_impedance_controller.h`, `include/franka_controllers/pseudo_inversion.h` | Cartesian impedance controller at 1 kHz |
| `cfg/compliance_param.cfg`, `msg/ZeroJacobian.msg`, `package.xml`, `LICENSE` | Package files |
| `src/joint_position_franka_controller.cpp`, `include/franka_controllers/joint_position_franka_controller.h`, `msg/JointFranka.msg` | Joint position controller with a velocity and acceleration limited ramp, used for homing |
| `CMakeLists.txt`, `franka_controllers_plugin.xml`, `config/franka_controllers.yaml` | Build and plugin files for the two controllers above |

## Build

```bash
cp -r third_party/franka_controllers ~/catkin_ws/src/
cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
```

The workspace also needs `franka_ros`. We used `franka_ros` 0.10.2 without modification, built from source against the libfranka version that matches the robot firmware.

## License

MIT, see `LICENSE`.
