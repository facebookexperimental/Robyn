"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.load = exports.loadPluginConfigs = exports.loadContext = void 0;
const utils_1 = require("@docusaurus/utils");
const path_1 = __importStar(require("path"));
const chalk_1 = __importDefault(require("chalk"));
const ssr_html_template_1 = __importDefault(require("../client/templates/ssr.html.template"));
const constants_1 = require("../constants");
const client_modules_1 = __importDefault(require("./client-modules"));
const config_1 = __importDefault(require("./config"));
const plugins_1 = require("./plugins");
const presets_1 = __importDefault(require("./presets"));
const routes_1 = __importDefault(require("./routes"));
const themes_1 = __importDefault(require("./themes"));
const html_tags_1 = require("./html-tags");
const versions_1 = require("./versions");
const duplicateRoutes_1 = require("./duplicateRoutes");
const i18n_1 = require("./i18n");
const translations_1 = require("./translations/translations");
const lodash_1 = require("lodash");
async function loadContext(siteDir, options = {}) {
    var _a;
    const { customOutDir, locale } = options;
    const generatedFilesDir = path_1.default.resolve(siteDir, constants_1.GENERATED_FILES_DIR_NAME);
    const initialSiteConfig = config_1.default(siteDir);
    const { ssrTemplate } = initialSiteConfig;
    const baseOutDir = customOutDir
        ? path_1.default.resolve(customOutDir)
        : path_1.default.resolve(siteDir, constants_1.BUILD_DIR_NAME);
    const i18n = await i18n_1.loadI18n(initialSiteConfig, { locale });
    const baseUrl = i18n_1.localizePath({
        path: initialSiteConfig.baseUrl,
        i18n,
        options,
        pathType: 'url',
    });
    const outDir = i18n_1.localizePath({
        path: baseOutDir,
        i18n,
        options,
        pathType: 'fs',
    });
    const siteConfig = Object.assign(Object.assign({}, initialSiteConfig), { baseUrl });
    const codeTranslationFileContent = (_a = (await translations_1.readCodeTranslationFileContent({
        siteDir,
        locale: i18n.currentLocale,
    }))) !== null && _a !== void 0 ? _a : {};
    // We only need key->message for code translations
    const codeTranslations = lodash_1.mapValues(codeTranslationFileContent, (value) => value.message);
    return {
        siteDir,
        generatedFilesDir,
        siteConfig,
        outDir,
        baseUrl,
        i18n,
        ssrTemplate,
        codeTranslations,
    };
}
exports.loadContext = loadContext;
function loadPluginConfigs(context) {
    const { plugins: presetPlugins, themes: presetThemes } = presets_1.default(context);
    const { siteConfig } = context;
    return [
        ...presetPlugins,
        ...presetThemes,
        // Site config should be the highest priority.
        ...(siteConfig.plugins || []),
        ...(siteConfig.themes || []),
    ];
}
exports.loadPluginConfigs = loadPluginConfigs;
async function load(siteDir, options = {}) {
    // Context.
    const context = await loadContext(siteDir, options);
    const { generatedFilesDir, siteConfig, outDir, baseUrl, i18n, ssrTemplate, codeTranslations, } = context;
    // Plugins.
    const pluginConfigs = loadPluginConfigs(context);
    const { plugins, pluginsRouteConfigs, globalData, themeConfigTranslated, } = await plugins_1.loadPlugins({
        pluginConfigs,
        context,
    });
    // Side-effect to replace the untranslated themeConfig by the translated one
    context.siteConfig.themeConfig = themeConfigTranslated;
    duplicateRoutes_1.handleDuplicateRoutes(pluginsRouteConfigs, siteConfig.onDuplicateRoutes);
    // Site config must be generated after plugins
    // We want the generated config to have been normalized by the plugins!
    const genSiteConfig = utils_1.generate(generatedFilesDir, constants_1.CONFIG_FILE_NAME, `export default ${JSON.stringify(siteConfig, null, 2)};`);
    // Themes.
    const fallbackTheme = path_1.default.resolve(__dirname, '../client/theme-fallback');
    const pluginThemes = plugins
        .map((plugin) => plugin.getThemePath && plugin.getThemePath())
        .filter((x) => Boolean(x));
    const userTheme = path_1.default.resolve(siteDir, constants_1.THEME_PATH);
    const alias = themes_1.default([fallbackTheme, ...pluginThemes], [userTheme]);
    // Make a fake plugin to:
    // - Resolve aliased theme components
    // - Inject scripts/stylesheets
    const { stylesheets = [], scripts = [], clientModules: siteConfigClientModules = [], } = siteConfig;
    plugins.push({
        name: 'docusaurus-bootstrap-plugin',
        options: {},
        version: { type: 'synthetic' },
        getClientModules() {
            return siteConfigClientModules;
        },
        configureWebpack: () => ({
            resolve: {
                alias,
            },
        }),
        injectHtmlTags: () => {
            const stylesheetsTags = stylesheets.map((source) => typeof source === 'string'
                ? `<link rel="stylesheet" href="${source}">`
                : {
                    tagName: 'link',
                    attributes: Object.assign({ rel: 'stylesheet' }, source),
                });
            const scriptsTags = scripts.map((source) => typeof source === 'string'
                ? `<script type="text/javascript" src="${source}"></script>`
                : {
                    tagName: 'script',
                    attributes: Object.assign({ type: 'text/javascript' }, source),
                });
            return {
                headTags: [...stylesheetsTags, ...scriptsTags],
            };
        },
    });
    // Load client modules.
    const clientModules = client_modules_1.default(plugins);
    const genClientModules = utils_1.generate(generatedFilesDir, 'client-modules.js', `export default [\n${clientModules
        // import() is async so we use require() because client modules can have
        // CSS and the order matters for loading CSS.
        // We need to JSON.stringify so that if its on windows, backslash are escaped.
        .map((module) => `  require(${JSON.stringify(module)}),`)
        .join('\n')}\n];\n`);
    // Load extra head & body html tags.
    const { headTags, preBodyTags, postBodyTags } = html_tags_1.loadHtmlTags(plugins);
    // Routing.
    const { registry, routesChunkNames, routesConfig, routesPaths, } = await routes_1.default(pluginsRouteConfigs, baseUrl);
    const genRegistry = utils_1.generate(generatedFilesDir, 'registry.js', `export default {
${Object.keys(registry)
        .sort()
        .map((key) => 
    // We need to JSON.stringify so that if its on windows, backslash are escaped.
    `  '${key}': [${registry[key].loader}, ${JSON.stringify(registry[key].modulePath)}, require.resolveWeak(${JSON.stringify(registry[key].modulePath)})],`)
        .join('\n')}};\n`);
    const genRoutesChunkNames = utils_1.generate(generatedFilesDir, 'routesChunkNames.json', JSON.stringify(routesChunkNames, null, 2));
    const genRoutes = utils_1.generate(generatedFilesDir, 'routes.js', routesConfig);
    const genGlobalData = utils_1.generate(generatedFilesDir, 'globalData.json', JSON.stringify(globalData, null, 2));
    const genI18n = utils_1.generate(generatedFilesDir, 'i18n.json', JSON.stringify(i18n, null, 2));
    const genCodeTranslations = utils_1.generate(generatedFilesDir, 'codeTranslations.json', JSON.stringify(codeTranslations, null, 2));
    // Version metadata.
    const siteMetadata = {
        docusaurusVersion: versions_1.getPackageJsonVersion(path_1.join(__dirname, '../../package.json')),
        siteVersion: versions_1.getPackageJsonVersion(path_1.join(siteDir, 'package.json')),
        pluginVersions: {},
    };
    plugins
        .filter(({ version: { type } }) => type !== 'synthetic')
        .forEach(({ name, version }) => {
        siteMetadata.pluginVersions[name] = version;
    });
    checkDocusaurusPackagesVersion(siteMetadata);
    const genSiteMetadata = utils_1.generate(generatedFilesDir, 'site-metadata.json', JSON.stringify(siteMetadata, null, 2));
    await Promise.all([
        genClientModules,
        genSiteConfig,
        genRegistry,
        genRoutesChunkNames,
        genRoutes,
        genGlobalData,
        genSiteMetadata,
        genI18n,
        genCodeTranslations,
    ]);
    const props = {
        siteConfig,
        siteDir,
        outDir,
        baseUrl,
        i18n,
        generatedFilesDir,
        routes: pluginsRouteConfigs,
        routesPaths,
        plugins,
        headTags,
        preBodyTags,
        postBodyTags,
        ssrTemplate: ssrTemplate || ssr_html_template_1.default,
        codeTranslations,
    };
    return props;
}
exports.load = load;
// We want all @docusaurus/* packages  to have the exact same version!
// See https://github.com/facebook/docusaurus/issues/3371
// See https://github.com/facebook/docusaurus/pull/3386
function checkDocusaurusPackagesVersion(siteMetadata) {
    const { docusaurusVersion } = siteMetadata;
    Object.entries(siteMetadata.pluginVersions).forEach(([plugin, versionInfo]) => {
        var _a;
        if (versionInfo.type === 'package' && ((_a = versionInfo.name) === null || _a === void 0 ? void 0 : _a.startsWith('@docusaurus/')) &&
            versionInfo.version !== docusaurusVersion) {
            // should we throw instead?
            // It still could work with different versions
            console.warn(chalk_1.default.red(`Bad ${plugin} version ${versionInfo.version}.\nAll official @docusaurus/* packages should have the exact same version as @docusaurus/core (${docusaurusVersion}).\nMaybe you want to check, or regenerate your yarn.lock or package-lock.json file?`));
        }
    });
}
