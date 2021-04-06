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
const fs_extra_1 = __importDefault(require("fs-extra"));
const path_1 = __importDefault(require("path"));
const chalk = require("chalk");
const constants_1 = require("../constants");
function removePath(fsPath) {
    return fs_extra_1.default
        .remove(path_1.default.join(fsPath))
        .then(() => {
        console.log(`${chalk.green(`Removing ${fsPath}`)}`);
    })
        .catch((err) => {
        console.error(`Could not remove ${fsPath}`);
        console.error(err);
    });
}
async function clear(siteDir) {
    return Promise.all([
        removePath(path_1.default.join(siteDir, constants_1.GENERATED_FILES_DIR_NAME)),
        removePath(path_1.default.join(siteDir, constants_1.BUILD_DIR_NAME)),
        removePath(path_1.default.join(siteDir, 'node_modules/.cache/cache-loader')),
    ]);
}
exports.default = clear;
