//
// Created by loy on 30-1-18.
//

#include "openpose_diagnostic_updater.h"

OpenposeDiagnosticUpdater::OpenposeDiagnosticUpdater()
{
    // TODO: Get expected frequency from parameter server
    service_diagnostic_ = new diagnostic_updater::FrequencyStatus(
            diagnostic_updater::FrequencyStatusParam(&expected_servicecall_frequency_, &expected_servicecall_frequency_, 0.1));

    add("General", this, &OpenposeDiagnosticUpdater::generalDiagnostics);
}

void OpenposeDiagnosticUpdater::tick()
{
    service_diagnostic_->tick();
    update(); // TODO: (Maybe) This needs to be called at some predefined interval, eg. on a ROS Timer
}

void OpenposeDiagnosticUpdater::generalDiagnostics(diagnostic_updater::DiagnosticStatusWrapper &stat)
{
    stat.summary(diagnostic_msgs::DiagnosticStatus::OK, "System OK.");
}