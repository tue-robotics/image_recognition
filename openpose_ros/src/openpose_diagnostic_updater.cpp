//
// Created by loy on 30-1-18.
//

#include "openpose_diagnostic_updater.h"

OpenposeDiagnosticUpdater::OpenposeDiagnosticUpdater()
{
    // TODO: Get expected frequency from parameter server
    expected_servicecall_frequency_ = 1.0;

    service_diagnostic_ = new diagnostic_updater::FrequencyStatus(
            diagnostic_updater::FrequencyStatusParam(&expected_servicecall_frequency_, &expected_servicecall_frequency_, 0.1));
    add(*service_diagnostic_);

    add("General", this, &OpenposeDiagnosticUpdater::generalDiagnostics);

    update_timer_ = nh_.createTimer(ros::Duration(1.0), &OpenposeDiagnosticUpdater::updateCallback, this);
}

void OpenposeDiagnosticUpdater::tick()
{
    service_diagnostic_->tick();
}

void OpenposeDiagnosticUpdater::generalDiagnostics(diagnostic_updater::DiagnosticStatusWrapper &stat)
{
    stat.summary(diagnostic_msgs::DiagnosticStatus::OK, "System OK.");
}

void OpenposeDiagnosticUpdater::updateCallback(const ros::TimerEvent& te)
{
    update();
}