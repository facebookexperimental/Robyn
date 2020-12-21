"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _clsx = _interopRequireDefault(require("clsx"));

var _SkipToContent = _interopRequireDefault(require("@theme/SkipToContent"));

var _AnnouncementBar = _interopRequireDefault(require("@theme/AnnouncementBar"));

var _Navbar = _interopRequireDefault(require("@theme/Navbar"));

var _Footer = _interopRequireDefault(require("@theme/Footer"));

var _LayoutProviders = _interopRequireDefault(require("@theme/LayoutProviders"));

var _LayoutHead = _interopRequireDefault(require("@theme/LayoutHead"));

var _useKeyboardNavigation = _interopRequireDefault(require("@theme/hooks/useKeyboardNavigation"));

require("./styles.css");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
function Layout(props) {
  const {
    children,
    noFooter,
    wrapperClassName
  } = props;
  (0, _useKeyboardNavigation.default)();
  return <_LayoutProviders.default>
      <_LayoutHead.default {...props} />

      <_SkipToContent.default />

      <_AnnouncementBar.default />

      <_Navbar.default />

      <div className={(0, _clsx.default)('main-wrapper', wrapperClassName)}>{children}</div>

      {!noFooter && <_Footer.default />}
    </_LayoutProviders.default>;
}

var _default = Layout;
exports.default = _default;