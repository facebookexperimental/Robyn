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
exports.loadPlugins = exports.sortConfig = void 0;
const utils_1 = require("@docusaurus/utils");
const fs_extra_1 = __importDefault(require("fs-extra"));
const path_1 = __importDefault(require("path"));
const init_1 = __importDefault(require("./init"));
const chalk_1 = __importDefault(require("chalk"));
const constants_1 = require("../../constants");
const lodash_1 = require("lodash");
const translations_1 = require("../translations/translations");
function sortConfig(routeConfigs) {
    // Sort the route config. This ensures that route with nested
    // routes is always placed last.
    routeConfigs.sort((a, b) => {
        if (a.routes && !b.routes) {
            return 1;
        }
        if (!a.routes && b.routes) {
            return -1;
        }
        // Higher priority get placed first.
        if (a.priority || b.priority) {
            const priorityA = a.priority || 0;
            const priorityB = b.priority || 0;
            const score = priorityB - priorityA;
            if (score !== 0) {
                return score;
            }
        }
        return a.path.localeCompare(b.path);
    });
    routeConfigs.forEach((routeConfig) => {
        var _a;
        (_a = routeConfig.routes) === null || _a === void 0 ? void 0 : _a.sort((a, b) => a.path.localeCompare(b.path));
    });
}
exports.sortConfig = sortConfig;
async function loadPlugins({ pluginConfigs, context, }) {
    // 1. Plugin Lifecycle - Initialization/Constructor.
    const plugins = init_1.default({
        pluginConfigs,
        context,
    });
    const contentLoadedPlugins = await Promise.all(plugins.map(async (plugin) => {
        const content = plugin.loadContent ? await plugin.loadContent() : null;
        return { plugin, content };
    }));
    const contentLoadedTranslatedPlugins = await Promise.all(contentLoadedPlugins.map(async (contentLoadedPlugin) => {
        var _a, _b, _c;
        const translationFiles = (_c = (await ((_b = (_a = contentLoadedPlugin.plugin) === null || _a === void 0 ? void 0 : _a.getTranslationFiles) === null || _b === void 0 ? void 0 : _b.call(_a)))) !== null && _c !== void 0 ? _c : [];
        const localizedTranslationFiles = await Promise.all(translationFiles.map((translationFile) => translations_1.localizePluginTranslationFile({
            locale: context.i18n.currentLocale,
            siteDir: context.siteDir,
            translationFile,
            plugin: contentLoadedPlugin.plugin,
        })));
        return Object.assign(Object.assign({}, contentLoadedPlugin), { translationFiles: localizedTranslationFiles });
    }));
    const allContent = lodash_1.chain(contentLoadedPlugins)
        .groupBy((item) => item.plugin.name)
        .mapValues((nameItems) => {
        return lodash_1.chain(nameItems)
            .groupBy((item) => { var _a; return (_a = item.plugin.options.id) !== null && _a !== void 0 ? _a : constants_1.DEFAULT_PLUGIN_ID; })
            .mapValues((idItems) => idItems[0].content)
            .value();
    })
        .value();
    // 3. Plugin Lifecycle - contentLoaded.
    const pluginsRouteConfigs = [];
    const globalData = {};
    await Promise.all(contentLoadedTranslatedPlugins.map(async ({ plugin, content, translationFiles }) => {
        var _a, _b, _c;
        if (!plugin.contentLoaded) {
            return;
        }
        const pluginId = (_a = plugin.options.id) !== null && _a !== void 0 ? _a : constants_1.DEFAULT_PLUGIN_ID;
        // plugins data files are namespaced by pluginName/pluginId
        const dataDirRoot = path_1.default.join(context.generatedFilesDir, plugin.name);
        const dataDir = path_1.default.join(dataDirRoot, pluginId);
        const addRoute = (config) => pluginsRouteConfigs.push(config);
        const createData = async (name, data) => {
            const modulePath = path_1.default.join(dataDir, name);
            await fs_extra_1.default.ensureDir(path_1.default.dirname(modulePath));
            await utils_1.generate(dataDir, name, data);
            return modulePath;
        };
        // the plugins global data are namespaced to avoid data conflicts:
        // - by plugin name
        // - by plugin id (allow using multiple instances of the same plugin)
        const setGlobalData = (data) => {
            var _a;
            globalData[plugin.name] = (_a = globalData[plugin.name]) !== null && _a !== void 0 ? _a : {};
            globalData[plugin.name][pluginId] = data;
        };
        const actions = {
            addRoute,
            createData,
            setGlobalData,
        };
        const translatedContent = (_c = (_b = plugin.translateContent) === null || _b === void 0 ? void 0 : _b.call(plugin, { content, translationFiles })) !== null && _c !== void 0 ? _c : content;
        await plugin.contentLoaded({
            content: translatedContent,
            actions,
            allContent,
        });
    }));
    // 4. Plugin Lifecycle - routesLoaded.
    // Currently plugins run lifecycle methods in parallel and are not order-dependent.
    // We could change this in future if there are plugins which need to
    // run in certain order or depend on others for data.
    await Promise.all(contentLoadedTranslatedPlugins.map(async ({ plugin }) => {
        if (!plugin.routesLoaded) {
            return null;
        }
        // TODO remove this deprecated lifecycle soon
        // deprecated since alpha-60
        console.error(chalk_1.default.red('plugin routesLoaded lifecycle is deprecated. If you think we should keep this lifecycle, please open a Github issue with your usecase'));
        return plugin.routesLoaded(pluginsRouteConfigs);
    }));
    // Sort the route config. This ensures that route with nested
    // routes are always placed last.
    sortConfig(pluginsRouteConfigs);
    // Apply each plugin one after the other to translate the theme config
    function translateThemeConfig(untranslatedThemeConfig) {
        return contentLoadedTranslatedPlugins.reduce((currentThemeConfig, { plugin, translationFiles }) => {
            var _a;
            const translatedThemeConfigSlice = (_a = plugin.translateThemeConfig) === null || _a === void 0 ? void 0 : _a.call(plugin, {
                themeConfig: currentThemeConfig,
                translationFiles,
            });
            return Object.assign(Object.assign({}, currentThemeConfig), translatedThemeConfigSlice);
        }, untranslatedThemeConfig);
    }
    return {
        plugins,
        pluginsRouteConfigs,
        globalData,
        themeConfigTranslated: translateThemeConfig(context.siteConfig.themeConfig),
    };
}
exports.loadPlugins = loadPlugins;
