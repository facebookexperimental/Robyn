/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import React from 'react';
import ExecutionEnvironment from './ExecutionEnvironment';
function BrowserOnly({ children, fallback, }) {
    if (!ExecutionEnvironment.canUseDOM || children == null) {
        return fallback || null;
    }
    return React.createElement(React.Fragment, null, children());
}
export default BrowserOnly;
