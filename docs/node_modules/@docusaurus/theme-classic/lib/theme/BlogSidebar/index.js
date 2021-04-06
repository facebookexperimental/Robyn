"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = BlogSidebar;

var _react = _interopRequireDefault(require("react"));

var _clsx = _interopRequireDefault(require("clsx"));

var _Link = _interopRequireDefault(require("@docusaurus/Link"));

var _stylesModule = _interopRequireDefault(require("./styles.module.css"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
function BlogSidebar({
  sidebar
}) {
  if (sidebar.items.length === 0) {
    return null;
  }

  return <div className={(0, _clsx.default)(_stylesModule.default.sidebar, 'thin-scrollbar')}>
      <h3 className={_stylesModule.default.sidebarItemTitle}>{sidebar.title}</h3>
      <ul className={_stylesModule.default.sidebarItemList}>
        {sidebar.items.map(item => {
        return <li key={item.permalink} className={_stylesModule.default.sidebarItem}>
              <_Link.default isNavLink to={item.permalink} className={_stylesModule.default.sidebarItemLink} activeClassName={_stylesModule.default.sidebarItemLinkActive}>
                {item.title}
              </_Link.default>
            </li>;
      })}
      </ul>
    </div>;
}