"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const loader_utils_1 = require("loader-utils");
const linkify_1 = require("./linkify");
const markdownLoader = function (source) {
    const fileString = source;
    const callback = this.async();
    const options = loader_utils_1.getOptions(this);
    return (callback && callback(null, linkify_1.linkify(fileString, this.resourcePath, options)));
};
exports.default = markdownLoader;
