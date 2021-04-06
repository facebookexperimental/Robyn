"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const storageKey = (pluginId) => `docs-preferred-version-${pluginId}`;
const DocsPreferredVersionStorage = {
    save: (pluginId, persistence, versionName) => {
        if (persistence === 'none') {
            // noop
        }
        else {
            window.localStorage.setItem(storageKey(pluginId), versionName);
        }
    },
    read: (pluginId, persistence) => {
        if (persistence === 'none') {
            return null;
        }
        else {
            return window.localStorage.getItem(storageKey(pluginId));
        }
    },
    clear: (pluginId, persistence) => {
        if (persistence === 'none') {
            // noop
        }
        else {
            window.localStorage.removeItem(storageKey(pluginId));
        }
    },
};
exports.default = DocsPreferredVersionStorage;
