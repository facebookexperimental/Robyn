"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = SearchMetadatas;

var _react = _interopRequireDefault(require("react"));

var _Head = _interopRequireDefault(require("@docusaurus/Head"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Note: we don't couple this to Algolia/DocSearch on purpose
// We may want to support other search engine plugins too
// Search plugins should swizzle/override this comp to add their behavior
function SearchMetadatas({
  locale,
  version,
  tag
}) {
  return <_Head.default>
      {locale && <meta name="docusaurus_locale" content={`${locale}`} />}
      {version && <meta name="docusaurus_version" content={version} />}
      {tag && <meta name="docusaurus_tag" content={tag} />}
    </_Head.default>;
}