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
const globby_1 = __importDefault(require("globby"));
const fs_extra_1 = __importDefault(require("fs-extra"));
const path_1 = __importDefault(require("path"));
const utils_1 = require("@docusaurus/utils");
const lodash_1 = require("lodash");
function themeAlias(themePath, addOriginalAlias) {
    if (!fs_extra_1.default.pathExistsSync(themePath)) {
        return {};
    }
    const themeComponentFiles = globby_1.default.sync(['**/*.{js,jsx,ts,tsx}'], {
        cwd: themePath,
    });
    // See https://github.com/facebook/docusaurus/pull/3922
    // ensure @theme/NavbarItem alias is created after @theme/NavbarItem/LocaleDropdown
    const sortedThemeComponentFiles = lodash_1.sortBy(themeComponentFiles, (file) => file.endsWith('/index.js'));
    const aliases = {};
    sortedThemeComponentFiles.forEach((relativeSource) => {
        const filePath = path_1.default.join(themePath, relativeSource);
        const fileName = utils_1.fileToPath(relativeSource);
        const aliasName = utils_1.posixPath(utils_1.normalizeUrl(['@theme', fileName]).replace(/\/$/, ''));
        aliases[aliasName] = filePath;
        if (addOriginalAlias) {
            // For swizzled components to access the original.
            const originalAliasName = utils_1.posixPath(utils_1.normalizeUrl(['@theme-original', fileName]).replace(/\/$/, ''));
            aliases[originalAliasName] = filePath;
        }
    });
    return aliases;
}
exports.default = themeAlias;
