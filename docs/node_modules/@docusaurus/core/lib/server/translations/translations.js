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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.localizePluginTranslationFile = exports.writePluginTranslations = exports.writeCodeTranslations = exports.readCodeTranslationFileContent = exports.getCodeTranslationsFilePath = exports.getTranslationsLocaleDirPath = exports.getTranslationsDirPath = exports.writeTranslationFileContent = exports.readTranslationFileContent = exports.ensureTranslationFileContent = void 0;
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const path_1 = __importDefault(require("path"));
const fs_extra_1 = __importDefault(require("fs-extra"));
const lodash_1 = require("lodash");
const utils_1 = require("@docusaurus/utils");
const Joi = __importStar(require("joi"));
const chalk_1 = __importDefault(require("chalk"));
const TranslationFileContentSchema = Joi.object()
    .pattern(Joi.string(), Joi.object({
    message: Joi.string().allow('').required(),
    description: Joi.string().optional(),
}))
    .required();
function ensureTranslationFileContent(content) {
    Joi.attempt(content, TranslationFileContentSchema, {
        abortEarly: false,
        allowUnknown: false,
        convert: false,
    });
}
exports.ensureTranslationFileContent = ensureTranslationFileContent;
async function readTranslationFileContent(filePath) {
    if (await fs_extra_1.default.pathExists(filePath)) {
        try {
            const content = JSON.parse(await fs_extra_1.default.readFile(filePath, 'utf8'));
            ensureTranslationFileContent(content);
            return content;
        }
        catch (e) {
            throw new Error(`Invalid translation file at path=${filePath}.\n${e.message}`);
        }
    }
    return undefined;
}
exports.readTranslationFileContent = readTranslationFileContent;
function mergeTranslationFileContent({ existingContent = {}, newContent, options, }) {
    // Apply messagePrefix to all messages
    const newContentTransformed = lodash_1.mapValues(newContent, (value) => {
        var _a;
        return (Object.assign(Object.assign({}, value), { message: `${(_a = options.messagePrefix) !== null && _a !== void 0 ? _a : ''}${value.message}` }));
    });
    const result = Object.assign({}, existingContent);
    // We only add missing keys here, we don't delete existing ones
    Object.entries(newContentTransformed).forEach(([key, { message, description }]) => {
        var _a, _b;
        result[key] = {
            // If the messages already exist, we don't override them (unless requested)
            message: options.override
                ? message
                : (_b = (_a = existingContent[key]) === null || _a === void 0 ? void 0 : _a.message) !== null && _b !== void 0 ? _b : message,
            description,
        };
    });
    return result;
}
async function writeTranslationFileContent({ filePath, content: newContent, options = {}, }) {
    const existingContent = await readTranslationFileContent(filePath);
    // Warn about potential legacy keys
    const unknownKeys = lodash_1.difference(Object.keys(existingContent !== null && existingContent !== void 0 ? existingContent : {}), Object.keys(newContent));
    if (unknownKeys.length > 0) {
        console.warn(chalk_1.default.yellow(`Some translation keys looks unknown to us in file ${filePath}
Maybe you should remove them?
- ${unknownKeys.join('\n- ')}`));
    }
    const mergedContent = mergeTranslationFileContent({
        existingContent,
        newContent,
        options,
    });
    // Avoid creating empty translation files
    if (Object.keys(mergedContent).length > 0) {
        console.log(`${Object.keys(mergedContent)
            .length.toString()
            .padStart(3, ' ')} translations written at ${path_1.default.relative(process.cwd(), filePath)}`);
        await fs_extra_1.default.ensureDir(path_1.default.dirname(filePath));
        await fs_extra_1.default.writeFile(filePath, JSON.stringify(mergedContent, null, 2));
    }
}
exports.writeTranslationFileContent = writeTranslationFileContent;
// should we make this configurable?
function getTranslationsDirPath(context) {
    return path_1.default.resolve(path_1.default.join(context.siteDir, `i18n`));
}
exports.getTranslationsDirPath = getTranslationsDirPath;
function getTranslationsLocaleDirPath(context) {
    return path_1.default.join(getTranslationsDirPath(context), context.locale);
}
exports.getTranslationsLocaleDirPath = getTranslationsLocaleDirPath;
function getCodeTranslationsFilePath(context) {
    return path_1.default.join(getTranslationsLocaleDirPath(context), 'code.json');
}
exports.getCodeTranslationsFilePath = getCodeTranslationsFilePath;
async function readCodeTranslationFileContent(context) {
    return readTranslationFileContent(getCodeTranslationsFilePath(context));
}
exports.readCodeTranslationFileContent = readCodeTranslationFileContent;
async function writeCodeTranslations(context, content, options) {
    return writeTranslationFileContent({
        filePath: getCodeTranslationsFilePath(context),
        content,
        options,
    });
}
exports.writeCodeTranslations = writeCodeTranslations;
// We ask users to not provide any extension on purpose:
// maybe some day we'll want to support multiple FS formats?
// (json/yaml/toml/xml...)
function addTranslationFileExtension(translationFilePath) {
    if (translationFilePath.endsWith('.json')) {
        throw new Error(`Translation file path does  not need to end  with .json, we addt the extension automatically. translationFilePath=${translationFilePath}`);
    }
    return `${translationFilePath}.json`;
}
function getPluginTranslationFilePath({ siteDir, plugin, locale, translationFilePath, }) {
    const dirPath = utils_1.getPluginI18nPath({
        siteDir,
        locale,
        pluginName: plugin.name,
        pluginId: plugin.options.id,
    });
    const filePath = addTranslationFileExtension(translationFilePath);
    return path_1.default.join(dirPath, filePath);
}
async function writePluginTranslations({ siteDir, plugin, locale, translationFile, options, }) {
    const filePath = getPluginTranslationFilePath({
        plugin,
        siteDir,
        locale,
        translationFilePath: translationFile.path,
    });
    await writeTranslationFileContent({
        filePath,
        content: translationFile.content,
        options,
    });
}
exports.writePluginTranslations = writePluginTranslations;
async function localizePluginTranslationFile({ siteDir, plugin, locale, translationFile, }) {
    const filePath = getPluginTranslationFilePath({
        plugin,
        siteDir,
        locale,
        translationFilePath: translationFile.path,
    });
    const localizedContent = await readTranslationFileContent(filePath);
    if (localizedContent) {
        // localized messages "override" default unlocalized messages
        return {
            path: translationFile.path,
            content: Object.assign(Object.assign({}, translationFile.content), localizedContent),
        };
    }
    else {
        return translationFile;
    }
}
exports.localizePluginTranslationFile = localizePluginTranslationFile;
