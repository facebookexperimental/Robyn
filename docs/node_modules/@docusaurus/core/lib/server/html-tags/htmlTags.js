"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const lodash_isplainobject_1 = __importDefault(require("lodash.isplainobject"));
const html_tags_1 = __importDefault(require("html-tags"));
const void_1 = __importDefault(require("html-tags/void"));
function assertIsHtmlTagObject(val) {
    if (!lodash_isplainobject_1.default(val)) {
        throw new Error(`"${val}" is not a valid HTML tag object`);
    }
    // @ts-expect-error: If tagName doesn't exist, it will throw.
    if (typeof val.tagName !== 'string') {
        throw new Error(`${JSON.stringify(val)} is not a valid HTML tag object. "tagName" must be defined as a string`);
    }
}
function htmlTagObjectToString(tagDefinition) {
    assertIsHtmlTagObject(tagDefinition);
    if (html_tags_1.default.indexOf(tagDefinition.tagName) === -1) {
        throw new Error(`Error loading ${JSON.stringify(tagDefinition)}, "${tagDefinition.tagName}" is not a valid HTML tags`);
    }
    const isVoidTag = void_1.default.indexOf(tagDefinition.tagName) !== -1;
    const tagAttributes = tagDefinition.attributes || {};
    const attributes = Object.keys(tagAttributes)
        .filter((attributeName) => tagAttributes[attributeName] !== false)
        .map((attributeName) => {
        if (tagAttributes[attributeName] === true) {
            return attributeName;
        }
        return `${attributeName}="${tagAttributes[attributeName]}"`;
    });
    return `<${[tagDefinition.tagName].concat(attributes).join(' ')}>${(!isVoidTag && tagDefinition.innerHTML) || ''}${isVoidTag ? '' : `</${tagDefinition.tagName}>`}`;
}
exports.default = htmlTagObjectToString;
