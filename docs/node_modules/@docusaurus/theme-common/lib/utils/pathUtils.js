"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.isSamePath = void 0;
// Compare the 2 paths, ignoring trailing /
exports.isSamePath = (path1, path2) => {
    const normalize = (pathname) => {
        return !pathname || (pathname === null || pathname === void 0 ? void 0 : pathname.endsWith('/')) ? pathname : `${pathname}/`;
    };
    return normalize(path1) === normalize(path2);
};
