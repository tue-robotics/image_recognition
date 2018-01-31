//
// Created by loy on 30-1-18.
//

#ifndef OPENPOSE_ROS_OPENPOSEDIAGNOSTICUPDATER_H
#define OPENPOSE_ROS_OPENPOSEDIAGNOSTICUPDATER_H

#include "diagnostic_updater/update_functions.h"
#include "diagnostic_updater/diagnostic_updater.h"
#include "diagnostic_updater/publisher.h"

class OpenposeDiagnosticUpdater : private diagnostic_updater::Updater
{
public:
    OpenposeDiagnosticUpdater();

    void tick();
    void generalDiagnostics(diagnostic_updater::DiagnosticStatusWrapper &stat);

private:
    double expected_servicecall_frequency_;
    diagnostic_updater::FrequencyStatus* service_diagnostic_;
};


#endif //OPENPOSE_ROS_OPENPOSEDIAGNOSTICUPDATER_H
