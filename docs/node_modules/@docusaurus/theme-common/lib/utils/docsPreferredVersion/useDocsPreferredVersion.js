"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.useDocsPreferredVersionByPluginId = exports.useDocsPreferredVersion = void 0;
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const react_1 = require("react");
const DocsPreferredVersionProvider_1 = require("./DocsPreferredVersionProvider");
const useDocs_1 = require("@theme/hooks/useDocs");
const constants_1 = require("@docusaurus/constants");
// TODO improve typing
// Note, the preferredVersion attribute will always be null before mount
function useDocsPreferredVersion(pluginId = constants_1.DEFAULT_PLUGIN_ID) {
    const docsData = useDocs_1.useDocsData(pluginId);
    const [state, api] = DocsPreferredVersionProvider_1.useDocsPreferredVersionContext();
    const { preferredVersionName } = state[pluginId];
    const preferredVersion = preferredVersionName
        ? docsData.versions.find((version) => version.name === preferredVersionName)
        : null;
    const savePreferredVersionName = react_1.useCallback((versionName) => {
        api.savePreferredVersion(pluginId, versionName);
    }, [api]);
    return { preferredVersion, savePreferredVersionName };
}
exports.useDocsPreferredVersion = useDocsPreferredVersion;
function useDocsPreferredVersionByPluginId() {
    const allDocsData = useDocs_1.useAllDocsData();
    const [state] = DocsPreferredVersionProvider_1.useDocsPreferredVersionContext();
    function getPluginIdPreferredVersion(pluginId) {
        const docsData = allDocsData[pluginId];
        const { preferredVersionName } = state[pluginId];
        return preferredVersionName
            ? docsData.versions.find((version) => version.name === preferredVersionName)
            : null;
    }
    const pluginIds = Object.keys(allDocsData);
    const result = {};
    pluginIds.forEach((pluginId) => {
        result[pluginId] = getPluginIdPreferredVersion(pluginId);
    });
    return result;
}
exports.useDocsPreferredVersionByPluginId = useDocsPreferredVersionByPluginId;
