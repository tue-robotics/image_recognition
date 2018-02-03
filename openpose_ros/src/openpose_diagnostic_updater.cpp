//
// Created by loy on 30-1-18.
//

#include "openpose_diagnostic_updater.h"

OpenposeDiagnosticUpdater::OpenposeDiagnosticUpdater()
{
    add("General", this, &OpenposeDiagnosticUpdater::generalDiagnostics);

    update_timer_ = nh_.createTimer(ros::Duration(1.0), &OpenposeDiagnosticUpdater::updateCallback, this);
}

void OpenposeDiagnosticUpdater::generalDiagnostics(diagnostic_updater::DiagnosticStatusWrapper &stat)
{
    stat.summary(diagnostic_msgs::DiagnosticStatus::OK, "System OK.");
}

void OpenposeDiagnosticUpdater::updateCallback(const ros::TimerEvent& te)
{
    update();
}