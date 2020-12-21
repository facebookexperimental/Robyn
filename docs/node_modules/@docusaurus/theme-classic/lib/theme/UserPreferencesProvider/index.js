"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _useTabGroupChoice = _interopRequireDefault(require("@theme/hooks/useTabGroupChoice"));

var _useAnnouncementBar = _interopRequireDefault(require("@theme/hooks/useAnnouncementBar"));

var _UserPreferencesContext = _interopRequireDefault(require("@theme/UserPreferencesContext"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
function UserPreferencesProvider(props) {
  const {
    tabGroupChoices,
    setTabGroupChoices
  } = (0, _useTabGroupChoice.default)();
  const {
    isAnnouncementBarClosed,
    closeAnnouncementBar
  } = (0, _useAnnouncementBar.default)();
  return <_UserPreferencesContext.default.Provider value={{
    tabGroupChoices,
    setTabGroupChoices,
    isAnnouncementBarClosed,
    closeAnnouncementBar
  }}>
      {props.children}
    </_UserPreferencesContext.default.Provider>;
}

var _default = UserPreferencesProvider;
exports.default = _default;