# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mark tests/ as a package.

test_bl4_final.py and test_swiglu_packed_onboarding.py import
``tests.independent_baseline``. Without this __init__.py the ``tests.``
namespace only resolves through pytest's rootdir sys.path insertion, which
is racy under pytest-xdist parallel collection (intermittent
"ImportError while importing test module" on worker startup). Making
tests/ a real package removes the race.
"""
