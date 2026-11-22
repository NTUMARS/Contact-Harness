/*
Refered to Source file:
  https://github.com/frankaemika/franka_ros/blob/develop/franka_example_controllers/src/joint_position_example_controller.cpp
*/

#include <franka_controllers/joint_position_franka_controller.h>

#include <algorithm>
#include <cmath>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_controllers {

bool JointPositionFrankaController::init(hardware_interface::RobotHW* robot_hardware,
                                         ros::NodeHandle& node_handle) {
  sub_joint_command_ = node_handle.subscribe(
      "joint_info", 20, &JointPositionFrankaController::jointCommandCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  position_joint_interface_ = robot_hardware->get<hardware_interface::PositionJointInterface>();
  
  if (position_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointPositionFrankaController: Error getting position joint interface from hardware!");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("JointPositionFrankaController: Could not parse joint names");
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("JointPositionFrankaController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }
  position_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      position_joint_handles_[i] = position_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& e) {
      ROS_ERROR_STREAM(
          "JointPositionFrankaController: Exception getting joint handles: " << e.what());
      return false;
    }
  }

  node_handle.getParam("filter_params", filter_params_);

  if (!node_handle.getParam("max_joint_velocity", max_joint_vel_)) {
    ROS_INFO_STREAM("JointPositionFrankaController: max_joint_velocity not found. Defaulting to "
                    << max_joint_vel_ << " rad/s");
  }
  max_joint_vel_ = std::max(0.001, max_joint_vel_);

  if (!node_handle.getParam("max_joint_acceleration", max_joint_accel_)) {
    ROS_INFO_STREAM("JointPositionFrankaController: max_joint_acceleration not found. Defaulting to "
                    << max_joint_accel_ << " rad/s^2");
  }
  max_joint_accel_ = std::max(0.01, max_joint_accel_);

  if (!node_handle.getParam("command_timeout", command_timeout_)) {
    ROS_INFO_STREAM("JointPositionFrankaController: command_timeout not found. Defaulting to "
                    << command_timeout_ << " s");
  }

  if (!node_handle.getParam("hybrid_switch_threshold", hybrid_switch_threshold_)) {
    ROS_INFO_STREAM("JointPositionFrankaController: hybrid_switch_threshold not found. Defaulting to "
                    << hybrid_switch_threshold_ << " rad");
  }
  hybrid_switch_threshold_ = std::max(0.0, hybrid_switch_threshold_);

  std::vector<double> target_positions;
  if (node_handle.getParam("/target_joint_positions", target_positions)) {
    if (target_positions.size() != 7) {
      ROS_ERROR("JointPositionFrankaController: /target_joint_positions has incorrect size");
      return false;
    }
    for (size_t i = 0; i < 7; ++i) {
      target_pose_[i] = target_positions[i];
    }
    has_target_pose_ = true;
  } else {
    ROS_INFO("JointPositionFrankaController: /target_joint_positions not set. Holding current joint pose until a command is received.");
  }

  JointCommand initial_command;
  initial_command.q = target_pose_;
  command_buffer_.writeFromNonRT(initial_command);

  return true;
}

void JointPositionFrankaController::starting(const ros::Time& time) {
  for (size_t i = 0; i < 7; ++i) {
    initial_pose_[i] = position_joint_handles_[i].getPosition();
    command_pose_[i] = initial_pose_[i];
    command_vel_[i] = 0.0;
    if (!has_target_pose_) {
      target_pose_[i] = initial_pose_[i];
    }
  }
  elapsed_time_ = ros::Duration(0.0);
  active_command_sequence_ = 0;
  command_received_ = false;
  last_command_time_ = time;

  JointCommand starting_command;
  starting_command.q = target_pose_;
  starting_command.sequence = has_target_pose_ ? 1 : 0;
  received_command_sequence_ = starting_command.sequence;
  command_buffer_.writeFromNonRT(starting_command);

  ROS_INFO("JointPositionFrankaController: started with max_joint_vel=%.3f rad/s, "
           "max_joint_accel=%.2f rad/s^2, command_timeout=%.2f s",
           max_joint_vel_, max_joint_accel_, command_timeout_);
}

void JointPositionFrankaController::update(const ros::Time& time,
                                           const ros::Duration& period) {
  elapsed_time_ += period;

  const JointCommand* command = command_buffer_.readFromRT();
  if (command != nullptr && command->sequence != active_command_sequence_) {
    active_command_sequence_ = command->sequence;
    target_pose_ = command->q;
    has_target_pose_ = true;
    last_command_time_ = time;
    command_received_ = true;
  }

  if (command_received_ && (time - last_command_time_).toSec() > command_timeout_) {
    ROS_WARN_ONCE(
        "JointPositionFrankaController: no command for %.1f s, holding position. "
        "This warning prints once; new commands will resume tracking.",
        (time - last_command_time_).toSec());
  }

  // Hybrid update: rate-limited ramp when far from target (smooth, bounded
  // velocity/acceleration — high-inertia joints stay quiet), first-order
  // low-pass when within hybrid_switch_threshold_ (asymptotic convergence
  // closes the last gap exactly).
  for (size_t i = 0; i < 7; ++i) {
    double error = target_pose_[i] - command_pose_[i];
    double abs_error = std::abs(error);

    if (abs_error > hybrid_switch_threshold_) {
      double v_decel = std::sqrt(2.0 * max_joint_accel_ * abs_error);
      double v_desired = std::copysign(std::min(max_joint_vel_, v_decel), error);

      double dv = v_desired - command_vel_[i];
      double max_dv = max_joint_accel_ * kControlPeriod;
      command_vel_[i] += std::clamp(dv, -max_dv, max_dv);
      command_vel_[i] = std::clamp(command_vel_[i], -max_joint_vel_, max_joint_vel_);

      command_pose_[i] += command_vel_[i] * kControlPeriod;
    } else {
      command_pose_[i] = filter_params_ * target_pose_[i]
                       + (1.0 - filter_params_) * command_pose_[i];
      // Bleed the ramp's velocity state so a future far target re-enters the
      // ramp from approximately zero velocity.
      command_vel_[i] *= (1.0 - filter_params_);
    }

    position_joint_handles_[i].setCommand(command_pose_[i]);
  }
}

void JointPositionFrankaController::jointCommandCallback(
    const franka_controllers::JointFranka& msg) {
  ROS_DEBUG_THROTTLE(1.0, "Received joint position command.");
  JointCommand command;
  for (size_t i = 0; i < 7; ++i) {
    command.q[i] = msg.joint_angle[i];
  }
  command.sequence = ++received_command_sequence_;
  command_buffer_.writeFromNonRT(command);
}

}  // namespace franka_controllers

PLUGINLIB_EXPORT_CLASS(franka_controllers::JointPositionFrankaController,
                       controller_interface::ControllerBase)
