"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _palenight = _interopRequireDefault(require("prism-react-renderer/themes/palenight"));

var _useThemeContext = _interopRequireDefault(require("@theme/hooks/useThemeContext"));

var _themeCommon = require("@docusaurus/theme-common");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const usePrismTheme = () => {
  const {
    prism
  } = (0, _themeCommon.useThemeConfig)();
  const {
    isDarkTheme
  } = (0, _useThemeContext.default)();
  const lightModeTheme = prism.theme || _palenight.default;
  const darkModeTheme = prism.darkTheme || lightModeTheme;
  const prismTheme = isDarkTheme ? darkModeTheme : lightModeTheme;
  return prismTheme;
};

var _default = usePrismTheme;
exports.default = _default;