"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getAllFinalRoutes = void 0;
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const lodash_flatmap_1 = __importDefault(require("lodash.flatmap"));
// Recursively get the final routes (routes with no subroutes)
function getAllFinalRoutes(routeConfig) {
    function getFinalRoutes(route) {
        return route.routes ? lodash_flatmap_1.default(route.routes, getFinalRoutes) : [route];
    }
    return lodash_flatmap_1.default(routeConfig, getFinalRoutes);
}
exports.getAllFinalRoutes = getAllFinalRoutes;
