"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _clsx = _interopRequireDefault(require("clsx"));

var _useTOCHighlight = _interopRequireDefault(require("@theme/hooks/useTOCHighlight"));

var _stylesModule = _interopRequireDefault(require("./styles.module.css"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const LINK_CLASS_NAME = 'table-of-contents__link';
const ACTIVE_LINK_CLASS_NAME = 'table-of-contents__link--active';
const TOP_OFFSET = 100;
/* eslint-disable jsx-a11y/control-has-associated-label */

function Headings({
  toc,
  isChild
}) {
  if (!toc.length) {
    return null;
  }

  return <ul className={isChild ? '' : 'table-of-contents table-of-contents__left-border'}>
      {toc.map(heading => <li key={heading.id}>
          <a href={`#${heading.id}`} className={LINK_CLASS_NAME} // Developer provided the HTML, so assume it's safe.
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{
        __html: heading.value
      }} />
          <Headings isChild toc={heading.children} />
        </li>)}
    </ul>;
}

function TOC({
  toc
}) {
  (0, _useTOCHighlight.default)(LINK_CLASS_NAME, ACTIVE_LINK_CLASS_NAME, TOP_OFFSET);
  return <div className={(0, _clsx.default)(_stylesModule.default.tableOfContents, 'thin-scrollbar')}>
      <Headings toc={toc} />
    </div>;
}

var _default = TOC;
exports.default = _default;