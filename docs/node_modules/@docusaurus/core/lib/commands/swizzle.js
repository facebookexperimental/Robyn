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
exports.getPluginNames = void 0;
const chalk = require("chalk");
const fs_extra_1 = __importDefault(require("fs-extra"));
const import_fresh_1 = __importDefault(require("import-fresh"));
const path_1 = __importDefault(require("path"));
const leven_1 = __importDefault(require("leven"));
const lodash_1 = require("lodash");
const constants_1 = require("../constants");
const server_1 = require("../server");
const init_1 = __importDefault(require("../server/plugins/init"));
const utils_validation_1 = require("@docusaurus/utils-validation");
function getPluginNames(plugins) {
    return plugins.map((plugin) => {
        const pluginPath = Array.isArray(plugin) ? plugin[0] : plugin;
        let packagePath = path_1.default.dirname(pluginPath);
        while (packagePath) {
            if (fs_extra_1.default.existsSync(path_1.default.join(packagePath, 'package.json'))) {
                break;
            }
            else {
                packagePath = path_1.default.dirname(packagePath);
            }
        }
        if (packagePath === '.') {
            return pluginPath;
        }
        return import_fresh_1.default(path_1.default.join(packagePath, 'package.json')).name;
    });
}
exports.getPluginNames = getPluginNames;
function walk(dir) {
    let results = [];
    const list = fs_extra_1.default.readdirSync(dir);
    list.forEach((file) => {
        const fullPath = path_1.default.join(dir, file);
        const stat = fs_extra_1.default.statSync(fullPath);
        if (stat && stat.isDirectory()) {
            results = results.concat(walk(fullPath));
        }
        else if (!/node_modules|.css|.d.ts|.d.map/.test(fullPath)) {
            results.push(fullPath);
        }
    });
    return results;
}
const formatComponentName = (componentName) => componentName
    .replace(/(\/|\\)index.(js|tsx|ts|jsx)/, '')
    .replace(/.(js|tsx|ts|jsx)/, '');
function readComponent(themePath) {
    return walk(themePath).map((filePath) => formatComponentName(path_1.default.relative(themePath, filePath)));
}
// load components from theme based on configurations
function getComponentName(themePath, plugin, danger) {
    var _a, _b;
    // support both commonjs and ES style exports
    const getSwizzleComponentList = (_b = (_a = plugin.default) === null || _a === void 0 ? void 0 : _a.getSwizzleComponentList) !== null && _b !== void 0 ? _b : plugin.getSwizzleComponentList;
    if (getSwizzleComponentList) {
        const allowedComponent = getSwizzleComponentList();
        if (danger) {
            return readComponent(themePath);
        }
        return allowedComponent;
    }
    return readComponent(themePath);
}
function themeComponents(themePath, plugin) {
    const components = colorCode(themePath, plugin);
    if (components.length === 0) {
        return `${chalk.red('No component to swizzle')}`;
    }
    return `
${chalk.cyan('Theme components available for swizzle')}

${chalk.green('green  =>')} recommended: lower breaking change risk
${chalk.red('red    =>')} internal: higher breaking change risk

${components.join('\n')}
`;
}
function formattedThemeNames(themeNames) {
    return `Themes available for swizzle:\n${themeNames.join('\n')}`;
}
function colorCode(themePath, plugin) {
    var _a, _b;
    // support both commonjs and ES style exports
    const getSwizzleComponentList = (_b = (_a = plugin.default) === null || _a === void 0 ? void 0 : _a.getSwizzleComponentList) !== null && _b !== void 0 ? _b : plugin.getSwizzleComponentList;
    const components = readComponent(themePath);
    const allowedComponent = getSwizzleComponentList
        ? getSwizzleComponentList()
        : [];
    const [greenComponents, redComponents] = lodash_1.partition(components, (comp) => allowedComponent.includes(comp));
    return [
        ...greenComponents.map((component) => chalk.green(component)),
        ...redComponents.map((component) => chalk.red(component)),
    ];
}
async function swizzle(siteDir, themeName, componentName, typescript, danger) {
    var _a, _b, _c, _d, _e;
    const context = await server_1.loadContext(siteDir);
    const pluginConfigs = server_1.loadPluginConfigs(context);
    const pluginNames = getPluginNames(pluginConfigs);
    const plugins = init_1.default({
        pluginConfigs,
        context,
    });
    const themeNames = pluginNames.filter((_, index) => typescript
        ? plugins[index].getTypeScriptThemePath
        : plugins[index].getThemePath);
    if (!themeName) {
        console.log(formattedThemeNames(themeNames));
        process.exit(1);
    }
    let pluginModule;
    try {
        pluginModule = import_fresh_1.default(themeName);
    }
    catch (_f) {
        let suggestion;
        themeNames.forEach((name) => {
            if (leven_1.default(name, themeName) < 4) {
                suggestion = name;
            }
        });
        chalk.red(`Theme ${themeName} not found. ${suggestion
            ? `Did you mean "${suggestion}" ?`
            : formattedThemeNames(themeNames)}`);
        process.exit(1);
    }
    const plugin = (_a = pluginModule.default) !== null && _a !== void 0 ? _a : pluginModule;
    const validateOptions = (_c = (_b = pluginModule.default) === null || _b === void 0 ? void 0 : _b.validateOptions) !== null && _c !== void 0 ? _c : pluginModule.validateOptions;
    let pluginOptions;
    const resolvedThemeName = require.resolve(themeName);
    // find the plugin from list of plugin and get options if specified
    pluginConfigs.forEach((pluginConfig) => {
        // plugin can be a [string], [string,object] or string.
        if (Array.isArray(pluginConfig)) {
            if (require.resolve(pluginConfig[0]) === resolvedThemeName) {
                if (pluginConfig.length === 2) {
                    const [, options] = pluginConfig;
                    pluginOptions = options;
                }
            }
        }
    });
    if (validateOptions) {
        pluginOptions = validateOptions({
            validate: utils_validation_1.normalizePluginOptions,
            options: pluginOptions,
        });
    }
    const pluginInstance = plugin(context, pluginOptions);
    const themePath = typescript
        ? (_d = pluginInstance.getTypeScriptThemePath) === null || _d === void 0 ? void 0 : _d.call(pluginInstance) : (_e = pluginInstance.getThemePath) === null || _e === void 0 ? void 0 : _e.call(pluginInstance);
    if (!themePath) {
        console.warn(chalk.yellow(typescript
            ? `${themeName} does not provide TypeScript theme code via "getTypeScriptThemePath()".`
            : `${themeName} does not provide any theme code.`));
        process.exit(1);
    }
    if (!componentName) {
        console.warn(themeComponents(themePath, pluginModule));
        process.exit(1);
    }
    const components = getComponentName(themePath, pluginModule, Boolean(danger));
    const formattedComponentName = formatComponentName(componentName);
    const isComponentExists = components.find((component) => component === formattedComponentName);
    let mostSuitableComponent = componentName;
    if (!isComponentExists) {
        let mostSuitableMatch = componentName;
        let score = formattedComponentName.length;
        components.forEach((component) => {
            if (component.toLowerCase() === formattedComponentName.toLowerCase()) {
                // may be components with same lowercase key, try to match closest component
                const currentScore = leven_1.default(formattedComponentName, component);
                if (currentScore < score) {
                    score = currentScore;
                    mostSuitableMatch = component;
                }
            }
        });
        if (mostSuitableMatch !== componentName) {
            mostSuitableComponent = mostSuitableMatch;
            console.log(chalk.red(`Component "${componentName}" doesn't exists.`), chalk.yellow(`"${mostSuitableComponent}" is swizzled instead of "${componentName}".`));
        }
    }
    let fromPath = path_1.default.join(themePath, mostSuitableComponent);
    let toPath = path_1.default.resolve(siteDir, constants_1.THEME_PATH, mostSuitableComponent);
    // Handle single TypeScript/JavaScript file only.
    // E.g: if <fromPath> does not exist, we try to swizzle <fromPath>.(ts|tsx|js) instead
    if (!fs_extra_1.default.existsSync(fromPath)) {
        if (fs_extra_1.default.existsSync(`${fromPath}.ts`)) {
            [fromPath, toPath] = [`${fromPath}.ts`, `${toPath}.ts`];
        }
        else if (fs_extra_1.default.existsSync(`${fromPath}.tsx`)) {
            [fromPath, toPath] = [`${fromPath}.tsx`, `${toPath}.tsx`];
        }
        else if (fs_extra_1.default.existsSync(`${fromPath}.js`)) {
            [fromPath, toPath] = [`${fromPath}.js`, `${toPath}.js`];
        }
        else {
            let suggestion;
            components.forEach((name) => {
                if (leven_1.default(name, mostSuitableComponent) < 3) {
                    suggestion = name;
                }
            });
            console.warn(chalk.red(`Component ${mostSuitableComponent} not found.`));
            console.warn(suggestion
                ? `Did you mean "${suggestion}"?`
                : `${themeComponents(themePath, pluginModule)}`);
            process.exit(1);
        }
    }
    if (!components.includes(mostSuitableComponent) && !danger) {
        console.warn(chalk.red(`${mostSuitableComponent} is an internal component, and have a higher breaking change probability. If you want to swizzle it, use the "--danger" flag.`));
        process.exit(1);
    }
    await fs_extra_1.default.copy(fromPath, toPath);
    const relativeDir = path_1.default.relative(process.cwd(), toPath);
    const fromMsg = chalk.blue(mostSuitableComponent
        ? `${themeName} ${chalk.yellow(mostSuitableComponent)}`
        : themeName);
    const toMsg = chalk.cyan(relativeDir);
    console.log(`\n${chalk.green('Success!')} Copied ${fromMsg} to ${toMsg}.\n`);
}
exports.default = swizzle;
