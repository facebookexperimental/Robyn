"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const server_1 = require("../server");
const init_1 = __importDefault(require("../server/plugins/init"));
const translations_1 = require("../server/translations/translations");
const translationsExtractor_1 = require("../server/translations/translationsExtractor");
const utils_1 = require("../webpack/utils");
async function writePluginTranslationFiles({ siteDir, plugin, locale, options, }) {
    if (plugin.getTranslationFiles) {
        const translationFiles = await plugin.getTranslationFiles();
        await Promise.all(translationFiles.map(async (translationFile) => {
            await translations_1.writePluginTranslations({
                siteDir,
                plugin,
                translationFile,
                locale,
                options,
            });
        }));
    }
}
async function writeTranslations(siteDir, options) {
    var _a;
    const context = await server_1.loadContext(siteDir);
    const pluginConfigs = server_1.loadPluginConfigs(context);
    const plugins = init_1.default({
        pluginConfigs,
        context,
    });
    const locale = (_a = options.locale) !== null && _a !== void 0 ? _a : context.i18n.defaultLocale;
    if (!context.i18n.locales.includes(locale)) {
        throw new Error(`Can't write-translation for locale that is not in the locale configuration file.
Unknown locale=[${locale}].
Available locales=[${context.i18n.locales.join(',')}]`);
    }
    const babelOptions = utils_1.getBabelOptions({
        isServer: true,
        babelOptions: utils_1.getCustomBabelConfigFilePath(siteDir),
    });
    const codeTranslations = await translationsExtractor_1.extractPluginsSourceCodeTranslations(plugins, babelOptions);
    await translations_1.writeCodeTranslations({ siteDir, locale }, codeTranslations, options);
    await Promise.all(plugins.map(async (plugin) => {
        await writePluginTranslationFiles({ siteDir, plugin, locale, options });
    }));
}
exports.default = writeTranslations;
