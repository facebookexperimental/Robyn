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
function useLocationHash(initialHash) {
  const [hash, setHash] = (0, _react.useState)(initialHash);
  (0, _react.useEffect)(() => {
    const handleHashChange = () => setHash(window.location.hash);

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);
  return [hash, setHash];
}

var _default = useLocationHash;
exports.default = _default;