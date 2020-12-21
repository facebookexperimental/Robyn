"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = _default;

var _react = _interopRequireDefault(require("react"));

var _reactToggle = _interopRequireDefault(require("react-toggle"));

var _themeCommon = require("@docusaurus/theme-common");

var _useDocusaurusContext = _interopRequireDefault(require("@docusaurus/useDocusaurusContext"));

var _clsx = _interopRequireDefault(require("clsx"));

var _stylesModule = _interopRequireDefault(require("./styles.module.css"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const Dark = ({
  icon,
  style
}) => <span className={(0, _clsx.default)(_stylesModule.default.toggle, _stylesModule.default.dark)} style={style}>
    {icon}
  </span>;

const Light = ({
  icon,
  style
}) => <span className={(0, _clsx.default)(_stylesModule.default.toggle, _stylesModule.default.light)} style={style}>
    {icon}
  </span>;

function _default(props) {
  const {
    colorMode: {
      switchConfig: {
        darkIcon,
        darkIconStyle,
        lightIcon,
        lightIconStyle
      }
    }
  } = (0, _themeCommon.useThemeConfig)();
  const {
    isClient
  } = (0, _useDocusaurusContext.default)();
  return <_reactToggle.default disabled={!isClient} icons={{
    checked: <Dark icon={darkIcon} style={darkIconStyle} />,
    unchecked: <Light icon={lightIcon} style={lightIconStyle} />
  }} {...props} />;
}