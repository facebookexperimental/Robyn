"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
Object.defineProperty(exports, "__esModule", { value: true });
// import {getOptions} from 'loader-utils';
const markdownLoader = function (fileString) {
    const callback = this.async();
    // const options = getOptions(this);
    // TODO provide additinal md processing here? like interlinking pages?
    // fileString = linkify(fileString)
    return callback && callback(null, fileString);
};
exports.default = markdownLoader;
