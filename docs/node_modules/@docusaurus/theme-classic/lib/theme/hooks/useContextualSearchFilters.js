"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = useContextualSearchFilters;

var _useDocs = require("@theme/hooks/useDocs");

var _themeCommon = require("@docusaurus/theme-common");

var _useDocusaurusContext = _interopRequireDefault(require("@docusaurus/useDocusaurusContext"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// We may want to support multiple search engines, don't couple that to Algolia/DocSearch
// Maybe users will want to use its own search engine solution
function useContextualSearchFilters() {
  const {
    i18n
  } = (0, _useDocusaurusContext.default)();
  const allDocsData = (0, _useDocs.useAllDocsData)();
  const activePluginAndVersion = (0, _useDocs.useActivePluginAndVersion)();
  const docsPreferredVersionByPluginId = (0, _themeCommon.useDocsPreferredVersionByPluginId)();

  function getDocPluginTags(pluginId) {
    var _activePluginAndVersi, _ref;

    const activeVersion = (activePluginAndVersion === null || activePluginAndVersion === void 0 ? void 0 : (_activePluginAndVersi = activePluginAndVersion.activePlugin) === null || _activePluginAndVersi === void 0 ? void 0 : _activePluginAndVersi.pluginId) === pluginId ? activePluginAndVersion.activeVersion : undefined;
    const preferredVersion = docsPreferredVersionByPluginId[pluginId];
    const latestVersion = allDocsData[pluginId].versions.find(v => v.isLast);
    const version = (_ref = activeVersion !== null && activeVersion !== void 0 ? activeVersion : preferredVersion) !== null && _ref !== void 0 ? _ref : latestVersion;
    return (0, _themeCommon.docVersionSearchTag)(pluginId, version.name);
  }

  const tags = [_themeCommon.DEFAULT_SEARCH_TAG, ...Object.keys(allDocsData).map(getDocPluginTags)];
  return {
    locale: i18n.currentLocale,
    tags
  };
}