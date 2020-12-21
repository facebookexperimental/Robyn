"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.useDocVersionSuggestions = exports.useActiveDocContext = exports.useActiveVersion = exports.useLatestVersion = exports.useVersions = exports.useActivePluginAndVersion = exports.useActivePlugin = exports.useDocsData = exports.useAllDocsData = void 0;
const router_1 = require("@docusaurus/router");
const useGlobalData_1 = require("@docusaurus/useGlobalData");
const docsClientUtils_1 = require("../../client/docsClientUtils");
exports.useAllDocsData = () => useGlobalData_1.useAllPluginInstancesData('docusaurus-plugin-content-docs');
exports.useDocsData = (pluginId) => useGlobalData_1.usePluginData('docusaurus-plugin-content-docs', pluginId);
exports.useActivePlugin = (options = {}) => {
    const data = exports.useAllDocsData();
    const { pathname } = router_1.useLocation();
    return docsClientUtils_1.getActivePlugin(data, pathname, options);
};
exports.useActivePluginAndVersion = (options = {}) => {
    const activePlugin = exports.useActivePlugin(options);
    const { pathname } = router_1.useLocation();
    if (activePlugin) {
        const activeVersion = docsClientUtils_1.getActiveVersion(activePlugin.pluginData, pathname);
        return {
            activePlugin,
            activeVersion,
        };
    }
    return undefined;
};
// versions are returned ordered (most recent first)
exports.useVersions = (pluginId) => {
    const data = exports.useDocsData(pluginId);
    return data.versions;
};
exports.useLatestVersion = (pluginId) => {
    const data = exports.useDocsData(pluginId);
    return docsClientUtils_1.getLatestVersion(data);
};
// Note: return undefined on doc-unrelated pages,
// because there's no version currently considered as active
exports.useActiveVersion = (pluginId) => {
    const data = exports.useDocsData(pluginId);
    const { pathname } = router_1.useLocation();
    return docsClientUtils_1.getActiveVersion(data, pathname);
};
exports.useActiveDocContext = (pluginId) => {
    const data = exports.useDocsData(pluginId);
    const { pathname } = router_1.useLocation();
    return docsClientUtils_1.getActiveDocContext(data, pathname);
};
// Useful to say "hey, you are not on the latest docs version, please switch"
exports.useDocVersionSuggestions = (pluginId) => {
    const data = exports.useDocsData(pluginId);
    const { pathname } = router_1.useLocation();
    return docsClientUtils_1.getDocVersionSuggestions(data, pathname);
};
