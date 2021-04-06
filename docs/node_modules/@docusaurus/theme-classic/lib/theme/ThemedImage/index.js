"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _clsx = _interopRequireDefault(require("clsx"));

var _useDocusaurusContext = _interopRequireDefault(require("@docusaurus/useDocusaurusContext"));

var _useThemeContext = _interopRequireDefault(require("@theme/hooks/useThemeContext"));

var _stylesModule = _interopRequireDefault(require("./styles.module.css"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const ThemedImage = props => {
  const {
    isClient
  } = (0, _useDocusaurusContext.default)();
  const {
    isDarkTheme
  } = (0, _useThemeContext.default)();
  const {
    sources,
    className,
    alt = '',
    ...propsRest
  } = props;
  const renderedSourceNames = isClient ? isDarkTheme ? ['dark'] : ['light'] : // We need to render both images on the server to avoid flash
  // See https://github.com/facebook/docusaurus/pull/3730
  ['light', 'dark'];
  return <>
      {renderedSourceNames.map(sourceName => <img key={sourceName} src={sources[sourceName]} alt={alt} className={(0, _clsx.default)(_stylesModule.default.themedImage, _stylesModule.default[`themedImage--${sourceName}`], className)} {...propsRest} />)}
    </>;
};

var _default = ThemedImage;
exports.default = _default;