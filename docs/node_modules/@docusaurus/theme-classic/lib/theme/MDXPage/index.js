"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _Layout = _interopRequireDefault(require("@theme/Layout"));

var _react2 = require("@mdx-js/react");

var _MDXComponents = _interopRequireDefault(require("@theme/MDXComponents"));

var _TOC = _interopRequireDefault(require("@theme/TOC"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
function MDXPage(props) {
  const {
    content: MDXPageContent
  } = props;
  const {
    frontMatter,
    metadata
  } = MDXPageContent;
  const {
    title,
    description,
    wrapperClassName,
    hide_table_of_contents: hideTableOfContents
  } = frontMatter;
  const {
    permalink
  } = metadata;
  return <_Layout.default title={title} description={description} permalink={permalink} wrapperClassName={wrapperClassName}>
      <main>
        <div className="container container--fluid">
          <div className="margin-vert--lg padding-vert--lg">
            <div className="row">
              <div className="col col--8 col--offset-2">
                <div className="container">
                  <_react2.MDXProvider components={_MDXComponents.default}>
                    <MDXPageContent />
                  </_react2.MDXProvider>
                </div>
              </div>
              {!hideTableOfContents && MDXPageContent.toc && <div className="col col--2">
                  <_TOC.default toc={MDXPageContent.toc} />
                </div>}
            </div>
          </div>
        </div>
      </main>
    </_Layout.default>;
}

var _default = MDXPage;
exports.default = _default;