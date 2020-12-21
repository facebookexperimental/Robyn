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
const chalk = require("chalk");
const copy_webpack_plugin_1 = __importDefault(require("copy-webpack-plugin"));
const fs_extra_1 = __importDefault(require("fs-extra"));
const path_1 = __importDefault(require("path"));
const react_loadable_ssr_addon_1 = __importDefault(require("react-loadable-ssr-addon"));
const webpack_bundle_analyzer_1 = require("webpack-bundle-analyzer");
const webpack_merge_1 = __importDefault(require("webpack-merge"));
const constants_1 = require("../constants");
const server_1 = require("../server");
const brokenLinks_1 = require("../server/brokenLinks");
const client_1 = __importDefault(require("../webpack/client"));
const server_2 = __importDefault(require("../webpack/server"));
const utils_1 = require("../webpack/utils");
const CleanWebpackPlugin_1 = __importDefault(require("../webpack/plugins/CleanWebpackPlugin"));
const i18n_1 = require("../server/i18n");
const utils_2 = require("@docusaurus/utils");
const config_1 = __importDefault(require("../server/config"));
async function build(siteDir, cliOptions = {}, forceTerminate = true) {
    async function tryToBuildLocale(locale, forceTerm) {
        try {
            const result = await buildLocale(siteDir, locale, cliOptions, forceTerm);
            console.log(chalk.green(`Site successfully built in locale=${locale}`));
            return result;
        }
        catch (e) {
            console.error(`error building locale=${locale}`);
            throw e;
        }
    }
    const i18n = await i18n_1.loadI18n(config_1.default(siteDir), {
        locale: cliOptions.locale,
    });
    if (cliOptions.locale) {
        return tryToBuildLocale(cliOptions.locale, forceTerminate);
    }
    else {
        if (i18n.locales.length > 1) {
            console.log(chalk.yellow(`\nSite will be built with all these locales:
- ${i18n.locales.join('\n- ')}\n`));
        }
        // We need the default locale to always be the 1st in the list
        // If we build it last, it would "erase" the localized sites built in subfolders
        const orderedLocales = [
            i18n.defaultLocale,
            ...i18n.locales.filter((locale) => locale !== i18n.defaultLocale),
        ];
        const results = await utils_2.mapAsyncSequencial(orderedLocales, (locale) => {
            const isLastLocale = i18n.locales.indexOf(locale) === i18n.locales.length - 1;
            // TODO check why we need forceTerminate
            const forceTerm = isLastLocale && forceTerminate;
            return tryToBuildLocale(locale, forceTerm);
        });
        return results[0];
    }
}
exports.default = build;
async function buildLocale(siteDir, locale, cliOptions = {}, forceTerminate = true) {
    process.env.BABEL_ENV = 'production';
    process.env.NODE_ENV = 'production';
    console.log(chalk.blue(`[${locale}] Creating an optimized production build...`));
    const props = await server_1.load(siteDir, {
        customOutDir: cliOptions.outDir,
        locale,
        localizePath: cliOptions.locale ? false : undefined,
    });
    // Apply user webpack config.
    const { outDir, generatedFilesDir, plugins, siteConfig: { baseUrl, onBrokenLinks }, routes, } = props;
    const clientManifestPath = path_1.default.join(generatedFilesDir, 'client-manifest.json');
    let clientConfig = webpack_merge_1.default(client_1.default(props, cliOptions.minify), {
        plugins: [
            // Remove/clean build folders before building bundles.
            new CleanWebpackPlugin_1.default({ verbose: false }),
            // Visualize size of webpack output files with an interactive zoomable treemap.
            cliOptions.bundleAnalyzer && new webpack_bundle_analyzer_1.BundleAnalyzerPlugin(),
            // Generate client manifests file that will be used for server bundle.
            new react_loadable_ssr_addon_1.default({
                filename: clientManifestPath,
            }),
        ].filter(Boolean),
    });
    const allCollectedLinks = {};
    let serverConfig = server_2.default({
        props,
        onLinksCollected: (staticPagePath, links) => {
            allCollectedLinks[staticPagePath] = links;
        },
    });
    const staticDir = path_1.default.resolve(siteDir, constants_1.STATIC_DIR_NAME);
    if (fs_extra_1.default.existsSync(staticDir)) {
        serverConfig = webpack_merge_1.default(serverConfig, {
            plugins: [
                new copy_webpack_plugin_1.default({
                    patterns: [
                        {
                            from: staticDir,
                            to: outDir,
                        },
                    ],
                }),
            ],
        });
    }
    // Plugin Lifecycle - configureWebpack.
    plugins.forEach((plugin) => {
        const { configureWebpack } = plugin;
        if (!configureWebpack) {
            return;
        }
        clientConfig = utils_1.applyConfigureWebpack(configureWebpack.bind(plugin), // The plugin lifecycle may reference `this`.
        clientConfig, false);
        serverConfig = utils_1.applyConfigureWebpack(configureWebpack.bind(plugin), // The plugin lifecycle may reference `this`.
        serverConfig, true);
    });
    // Make sure generated client-manifest is cleaned first so we don't reuse
    // the one from previous builds.
    if (fs_extra_1.default.existsSync(clientManifestPath)) {
        fs_extra_1.default.unlinkSync(clientManifestPath);
    }
    // Run webpack to build JS bundle (client) and static html files (server).
    await utils_1.compile([clientConfig, serverConfig]);
    // Remove server.bundle.js because it is not needed.
    if (serverConfig.output &&
        serverConfig.output.filename &&
        typeof serverConfig.output.filename === 'string') {
        const serverBundle = path_1.default.join(outDir, serverConfig.output.filename);
        fs_extra_1.default.pathExists(serverBundle).then((exist) => {
            if (exist) {
                fs_extra_1.default.unlink(serverBundle);
            }
        });
    }
    // Plugin Lifecycle - postBuild.
    await Promise.all(plugins.map(async (plugin) => {
        if (!plugin.postBuild) {
            return;
        }
        await plugin.postBuild(props);
    }));
    await brokenLinks_1.handleBrokenLinks({
        allCollectedLinks,
        routes,
        onBrokenLinks,
        outDir,
        baseUrl,
    });
    const relativeDir = path_1.default.relative(process.cwd(), outDir);
    console.log(`\n${chalk.green('Success!')} Generated static files in ${chalk.cyan(relativeDir)}. Use ${chalk.greenBright('`npm run serve`')} to test your build locally.\n`);
    if (forceTerminate && !cliOptions.bundleAnalyzer) {
        process.exit(0);
    }
    return outDir;
}
