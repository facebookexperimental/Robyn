"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = require("react");

var _UserPreferencesContext = _interopRequireDefault(require("@theme/UserPreferencesContext"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
function useUserPreferencesContext() {
  const context = (0, _react.useContext)(_UserPreferencesContext.default);

  if (context == null) {
    throw new Error('`useUserPreferencesContext` is used outside of `Layout` Component.');
  }

  return context;
}

var _default = useUserPreferencesContext;
exports.default = _default;