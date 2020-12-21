"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ensureUniquePluginInstanceIds = void 0;
const lodash_1 = require("lodash");
const constants_1 = require("../../constants");
// It is forbidden to have 2 plugins of the same name sharing the same id
// this is required to support multi-instance plugins without conflict
function ensureUniquePluginInstanceIds(plugins) {
    const pluginsByName = lodash_1.groupBy(plugins, (p) => p.name);
    Object.entries(pluginsByName).forEach(([pluginName, pluginInstances]) => {
        const pluginInstancesById = lodash_1.groupBy(pluginInstances, (p) => { var _a; return (_a = p.options.id) !== null && _a !== void 0 ? _a : constants_1.DEFAULT_PLUGIN_ID; });
        Object.entries(pluginInstancesById).forEach(([pluginId, pluginInstancesWithId]) => {
            if (pluginInstancesWithId.length !== 1) {
                throw new Error(`Plugin ${pluginName} is used ${pluginInstancesWithId.length} times with id=${pluginId}.\nTo use the same plugin multiple times on a Docusaurus site, you need to assign a unique id to each plugin instance.`);
            }
        });
    });
}
exports.ensureUniquePluginInstanceIds = ensureUniquePluginInstanceIds;
