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
const alias_1 = __importDefault(require("./alias"));
function buildThemeAliases(themeAliases, aliases = {}) {
    Object.keys(themeAliases).forEach((aliasKey) => {
        if (aliasKey in aliases) {
            const componentName = aliasKey.substring(aliasKey.indexOf('/') + 1);
            // eslint-disable-next-line no-param-reassign
            aliases[`@theme-init/${componentName}`] = aliases[aliasKey];
        }
        // eslint-disable-next-line no-param-reassign
        aliases[aliasKey] = themeAliases[aliasKey];
    });
    return aliases;
}
function loadThemeAlias(themePaths, userThemePaths = []) {
    let aliases = {};
    themePaths.forEach((themePath) => {
        const themeAliases = alias_1.default(themePath, true);
        aliases = Object.assign(Object.assign({}, aliases), buildThemeAliases(themeAliases, aliases));
    });
    userThemePaths.forEach((themePath) => {
        const userThemeAliases = alias_1.default(themePath, false);
        aliases = Object.assign(Object.assign({}, aliases), buildThemeAliases(userThemeAliases, aliases));
    });
    return aliases;
}
exports.default = loadThemeAlias;
