"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _useTheme = _interopRequireDefault(require("@theme/hooks/useTheme"));

var _ThemeContext = _interopRequireDefault(require("@theme/ThemeContext"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
function ThemeProvider(props) {
  const {
    isDarkTheme,
    setLightTheme,
    setDarkTheme
  } = (0, _useTheme.default)();
  return <_ThemeContext.default.Provider value={{
    isDarkTheme,
    setLightTheme,
    setDarkTheme
  }}>
      {props.children}
    </_ThemeContext.default.Provider>;
}

var _default = ThemeProvider;
exports.default = _default;