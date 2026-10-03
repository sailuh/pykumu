# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""JVM and Tetrad JAR initialization.

This module provides the `start` function to boot the JVM and load
the Tetrad JAR via JPype. It must be called once before using any
other api module that accesses Tetrad Java classes.
"""

import os
import warnings

import jpype
import jpype.imports


def tetrad_jvm_start(jar_path, jvm_args=None):
    """Start the JVM and load the Tetrad JAR.

    Must be called once before using any api module that accesses
    Tetrad Java classes (data, score, algorithm, etc.).

    :param jar_path: Path to the ``tetrad-current.jar`` file.
    :param jvm_args: Optional list of extra JVM arguments
        (e.g. ``["-Xmx8g"]`` to increase heap memory).
    :raises FileNotFoundError: If *jar_path* does not exist.
    """
    jar_path = os.path.abspath(jar_path)
    if not os.path.isfile(jar_path):
        raise FileNotFoundError(f"Tetrad JAR not found: {jar_path}")

    if jpype.isJVMStarted():
        warnings.warn(
            "JVM is already running. tetrad.start() has no effect after the "
            "first call — restart the Python process to change the JAR.",
            stacklevel=2,
        )
        return

    args = [jpype.getDefaultJVMPath()]
    if jvm_args:
        args.extend(jvm_args)

    jpype.startJVM(*args, classpath=[jar_path])
