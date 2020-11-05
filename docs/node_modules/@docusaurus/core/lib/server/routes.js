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
const utils_1 = require("@docusaurus/utils");
const lodash_has_1 = __importDefault(require("lodash.has"));
const lodash_isplainobject_1 = __importDefault(require("lodash.isplainobject"));
const lodash_isstring_1 = __importDefault(require("lodash.isstring"));
const querystring_1 = require("querystring");
const createRouteCodeString = ({ routePath, routeHash, exact, subroutesCodeStrings, }) => {
    const str = `{
  path: '${routePath}',
  component: ComponentCreator('${routePath}','${routeHash}'),
  ${exact ? `exact: true,` : ''}
${subroutesCodeStrings
        ? `  routes: [
${utils_1.removeSuffix(subroutesCodeStrings.join(',\n'), ',\n')},
]
`
        : ''}}`;
    return str;
};
const NotFoundRouteCode = `{
  path: '*',
  component: ComponentCreator('*')
}`;
const RoutesImportsCode = [
    `import React from 'react';`,
    `import ComponentCreator from '@docusaurus/ComponentCreator';`,
].join('\n');
function isModule(value) {
    if (lodash_isstring_1.default(value)) {
        return true;
    }
    if (lodash_isplainobject_1.default(value) && lodash_has_1.default(value, '__import') && lodash_has_1.default(value, 'path')) {
        return true;
    }
    return false;
}
function getModulePath(target) {
    if (typeof target === 'string') {
        return target;
    }
    const queryStr = target.query ? `?${querystring_1.stringify(target.query)}` : '';
    return `${target.path}${queryStr}`;
}
async function loadRoutes(pluginsRouteConfigs, baseUrl) {
    const registry = {};
    const routesPaths = [utils_1.normalizeUrl([baseUrl, '404.html'])];
    const routesChunkNames = {};
    // This is the higher level overview of route code generation.
    function generateRouteCode(routeConfig) {
        const { path: routePath, component, modules = {}, routes: subroutes, exact, } = routeConfig;
        if (!lodash_isstring_1.default(routePath) || !component) {
            throw new Error(`Invalid routeConfig (Path must be a string and component is required) \n${JSON.stringify(routeConfig)}`);
        }
        // Collect all page paths for injecting it later in the plugin lifecycle
        // This is useful for plugins like sitemaps, redirects etc...
        // If a route has subroutes, it is not necessarily a valid page path (more likely to be a wrapper)
        if (!subroutes) {
            routesPaths.push(routePath);
        }
        // We hash the route to generate the key, because 2 routes can conflict with
        // each others if they have the same path, ex: parent=/docs, child=/docs
        // see https://github.com/facebook/docusaurus/issues/2917
        const routeHash = utils_1.simpleHash(JSON.stringify(routeConfig), 3);
        const chunkNamesKey = `${routePath}-${routeHash}`;
        routesChunkNames[chunkNamesKey] = Object.assign(Object.assign({}, genRouteChunkNames(registry, { component }, 'component', component)), genRouteChunkNames(registry, modules, 'module', routePath));
        return createRouteCodeString({
            routePath: routeConfig.path,
            routeHash,
            exact,
            subroutesCodeStrings: subroutes === null || subroutes === void 0 ? void 0 : subroutes.map(generateRouteCode),
        });
    }
    const routesConfig = `
${RoutesImportsCode}
export default [
${pluginsRouteConfigs.map(generateRouteCode).join(',\n')},
${NotFoundRouteCode}
];\n`;
    return {
        registry,
        routesConfig,
        routesChunkNames,
        routesPaths,
    };
}
exports.default = loadRoutes;
function genRouteChunkNames(
// TODO instead of passing a mutating the registry, return a registry slice?
registry, value, prefix, name) {
    if (!value) {
        return null;
    }
    if (Array.isArray(value)) {
        return value.map((val, index) => genRouteChunkNames(registry, val, `${index}`, name));
    }
    if (isModule(value)) {
        const modulePath = getModulePath(value);
        const chunkName = utils_1.genChunkName(modulePath, prefix, name);
        // We need to JSON.stringify so that if its on windows, backslashes are escaped.
        const loader = `() => import(/* webpackChunkName: '${chunkName}' */ ${JSON.stringify(modulePath)})`;
        registry[chunkName] = {
            loader,
            modulePath,
        };
        return chunkName;
    }
    const newValue = {};
    Object.keys(value).forEach((key) => {
        newValue[key] = genRouteChunkNames(registry, value[key], key, name);
    });
    return newValue;
}
