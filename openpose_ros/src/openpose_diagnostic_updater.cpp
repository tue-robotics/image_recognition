//
// Created by loy on 30-1-18.
//

#include "openpose_diagnostic_updater.h"

OpenposeDiagnosticUpdater::OpenposeDiagnosticUpdater()
{
    // TODO: Get expected frequency from parameter server
    service_diagnostic_ = new diagnostic_updater::FrequencyStatus(
            diagnostic_updater::FrequencyStatusParam(&expected_servicecall_frequency_, &expected_servicecall_frequency_, 0.1));
}

void OpenposeDiagnosticUpdater::tick()
{
    service_diagnostic_->tick();
    update();
}