"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.handleDuplicateRoutes = exports.getDuplicateRoutesMessage = exports.getAllDuplicateRoutes = void 0;
const utils_1 = require("@docusaurus/utils");
const utils_2 = require("./utils");
function getAllDuplicateRoutes(pluginsRouteConfigs) {
    const allRoutes = utils_2.getAllFinalRoutes(pluginsRouteConfigs).map((routeConfig) => routeConfig.path);
    const seenRoutes = {};
    return allRoutes.filter((route) => {
        if (Object.prototype.hasOwnProperty.call(seenRoutes, route)) {
            return true;
        }
        else {
            seenRoutes[route] = true;
            return false;
        }
    });
}
exports.getAllDuplicateRoutes = getAllDuplicateRoutes;
function getDuplicateRoutesMessage(allDuplicateRoutes) {
    const message = allDuplicateRoutes
        .map((duplicateRoute) => `Attempting to create page at ${duplicateRoute}, but a page already exists at this route`)
        .join('\n');
    return message;
}
exports.getDuplicateRoutesMessage = getDuplicateRoutesMessage;
function handleDuplicateRoutes(pluginsRouteConfigs, onDuplicateRoutes) {
    if (onDuplicateRoutes === 'ignore') {
        return;
    }
    const duplicatePaths = getAllDuplicateRoutes(pluginsRouteConfigs);
    const message = getDuplicateRoutesMessage(duplicatePaths);
    if (message) {
        const finalMessage = `Duplicate routes found!\n${message}\nThis could lead to non-deterministic routing behavior`;
        utils_1.reportMessage(finalMessage, onDuplicateRoutes);
    }
}
exports.handleDuplicateRoutes = handleDuplicateRoutes;
