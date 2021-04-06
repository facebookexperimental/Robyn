"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.isDocsPluginEnabled = void 0;
const useDocs_1 = require("@theme/hooks/useDocs");
// TODO not ideal, see also "useDocs"
exports.isDocsPluginEnabled = !!useDocs_1.useAllDocsData;
