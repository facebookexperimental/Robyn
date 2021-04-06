"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PathnameSchema = exports.URISchema = exports.AdmonitionsSchema = exports.RehypePluginsSchema = exports.RemarkPluginsSchema = exports.PluginIdSchema = void 0;
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const Joi = __importStar(require("joi"));
const utils_1 = require("@docusaurus/utils");
exports.PluginIdSchema = Joi.string()
    .regex(/^[a-zA-Z_\-]+$/)
    // duplicate core constant, otherwise cyclic dependency is created :(
    .default('default');
const MarkdownPluginsSchema = Joi.array()
    .items(Joi.array().ordered(Joi.function().required(), Joi.object().required()), Joi.function(), Joi.object())
    .default([]);
exports.RemarkPluginsSchema = MarkdownPluginsSchema;
exports.RehypePluginsSchema = MarkdownPluginsSchema;
exports.AdmonitionsSchema = Joi.object().default({});
exports.URISchema = Joi.alternatives(Joi.string().uri({ allowRelative: true }), Joi.custom((val, helpers) => {
    try {
        const url = new URL(val);
        if (url) {
            return val;
        }
        else {
            return helpers.error('any.invalid');
        }
    }
    catch (_a) {
        return helpers.error('any.invalid');
    }
}));
exports.PathnameSchema = Joi.string()
    .custom((val) => {
    if (!utils_1.isValidPathname(val)) {
        throw new Error();
    }
    else {
        return val;
    }
})
    .message('{{#label}} is not a valid pathname. Pathname should start with / and not contain any domain or query string');
