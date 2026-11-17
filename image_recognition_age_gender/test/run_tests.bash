#!/bin/bash
python3 -m pytest -vv "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
