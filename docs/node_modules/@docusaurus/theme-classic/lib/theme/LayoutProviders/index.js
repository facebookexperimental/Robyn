"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = LayoutProviders;

var _react = _interopRequireDefault(require("react"));

var _ThemeProvider = _interopRequireDefault(require("@theme/ThemeProvider"));

var _UserPreferencesProvider = _interopRequireDefault(require("@theme/UserPreferencesProvider"));

var _themeCommon = require("@docusaurus/theme-common");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
function LayoutProviders({
  children
}) {
  return <_ThemeProvider.default>
      <_UserPreferencesProvider.default>
        <_themeCommon.DocsPreferredVersionContextProvider>
          {children}
        </_themeCommon.DocsPreferredVersionContextProvider>
      </_UserPreferencesProvider.default>
    </_ThemeProvider.default>;
}