#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_buffer.h>
#include <ros/node_handle.h>
#include <ros/subscriber.h>
#include <ros/time.h>
#include <franka_controllers/JointFranka.h>

namespace franka_controllers {

class JointPositionFrankaController : public controller_interface::MultiInterfaceController<
                                          hardware_interface::PositionJointInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  struct JointCommand {
    std::array<double, 7> q{};
    uint64_t sequence{0};
  };

  void jointCommandCallback(const franka_controllers::JointFranka& msg);

  static constexpr double kControlPeriod = 0.001;  // 1 kHz

  hardware_interface::PositionJointInterface* position_joint_interface_;
  std::vector<hardware_interface::JointHandle> position_joint_handles_;
  ros::Subscriber sub_joint_command_;
  ros::Duration elapsed_time_;
  std::array<double, 7> initial_pose_{};
  std::array<double, 7> target_pose_{};
  std::array<double, 7> command_pose_{};
  std::array<double, 7> command_vel_{};   // current velocity of the command ramp [rad/s]
  bool has_target_pose_{false};
  double filter_params_{0.005};           // first-order low-pass blend toward target (near-zone)
  double max_joint_vel_{0.15};            // max command velocity [rad/s] (far-zone ramp)
  double max_joint_accel_{2.5};           // max command acceleration [rad/s²] (far-zone ramp)
  double hybrid_switch_threshold_{0.01};  // |error| below this switches from ramp to low-pass [rad]
  double command_timeout_{0.5};           // hold position after this many seconds without a command

  realtime_tools::RealtimeBuffer<JointCommand> command_buffer_;
  uint64_t active_command_sequence_{0};
  uint64_t received_command_sequence_{0};
  ros::Time last_command_time_;
  bool command_received_{false};
};

}  // namespace franka_controllers
