"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const blogUtils_1 = require("./blogUtils");
const loader_utils_1 = require("loader-utils");
const markdownLoader = function (source) {
    const filePath = this.resourcePath;
    const fileContent = source;
    const callback = this.async();
    const markdownLoaderOptions = loader_utils_1.getOptions(this);
    // Linkify blog posts
    let finalContent = blogUtils_1.linkify(Object.assign({ fileContent,
        filePath }, markdownLoaderOptions));
    // Truncate content if requested (e.g: file.md?truncated=true).
    const truncated = this.resourceQuery
        ? loader_utils_1.parseQuery(this.resourceQuery).truncated
        : undefined;
    if (truncated) {
        finalContent = blogUtils_1.truncate(finalContent, markdownLoaderOptions.truncateMarker);
    }
    return callback && callback(null, finalContent);
};
exports.default = markdownLoader;
