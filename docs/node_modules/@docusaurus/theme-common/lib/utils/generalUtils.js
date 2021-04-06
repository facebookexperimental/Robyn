"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.useTitleFormatter = void 0;
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const useDocusaurusContext_1 = __importDefault(require("@docusaurus/useDocusaurusContext"));
exports.useTitleFormatter = (title) => {
    const { siteConfig = {} } = useDocusaurusContext_1.default();
    const { title: siteTitle, titleDelimiter = '|' } = siteConfig;
    return title && title.trim().length
        ? `${title.trim()} ${titleDelimiter} ${siteTitle}`
        : siteTitle;
};
