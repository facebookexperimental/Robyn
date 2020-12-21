"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const module_1 = __importDefault(require("module"));
const path_1 = require("path");
const import_fresh_1 = __importDefault(require("import-fresh"));
const constants_1 = require("../../constants");
function loadPresets(context) {
    // We need to resolve plugins from the perspective of the siteDir, since the siteDir's package.json
    // declares the dependency on these plugins.
    // We need to fallback to createRequireFromPath since createRequire is only available in node v12.
    // See: https://nodejs.org/api/modules.html#modules_module_createrequire_filename
    const createRequire = module_1.default.createRequire || module_1.default.createRequireFromPath;
    const pluginRequire = createRequire(path_1.join(context.siteDir, constants_1.CONFIG_FILE_NAME));
    const presets = (context.siteConfig || {}).presets || [];
    const unflatPlugins = [];
    const unflatThemes = [];
    presets.forEach((presetItem) => {
        let presetModuleImport;
        let presetOptions = {};
        if (typeof presetItem === 'string') {
            presetModuleImport = presetItem;
        }
        else if (Array.isArray(presetItem)) {
            [presetModuleImport, presetOptions = {}] = presetItem;
        }
        else {
            throw new Error('Invalid presets format detected in config.');
        }
        const presetModule = import_fresh_1.default(pluginRequire.resolve(presetModuleImport));
        const preset = (presetModule.default || presetModule)(context, presetOptions);
        if (preset.plugins) {
            unflatPlugins.push(preset.plugins);
        }
        if (preset.themes) {
            unflatThemes.push(preset.themes);
        }
    });
    return {
        plugins: [].concat(...unflatPlugins).filter(Boolean),
        themes: [].concat(...unflatThemes).filter(Boolean),
    };
}
exports.default = loadPresets;
