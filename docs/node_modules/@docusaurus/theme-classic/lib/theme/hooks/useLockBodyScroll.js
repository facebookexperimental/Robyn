"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = require("react");

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
function useLockBodyScroll(lock = true) {
  (0, _react.useEffect)(() => {
    document.body.style.overflow = lock ? 'hidden' : 'visible';
    return () => {
      document.body.style.overflow = 'visible';
    };
  }, [lock]);
}

var _default = useLockBodyScroll;
exports.default = _default;