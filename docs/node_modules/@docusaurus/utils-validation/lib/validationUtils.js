"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.normalizeThemeConfig = exports.normalizePluginOptions = exports.logValidationBugReportHint = exports.isValidationDisabledEscapeHatch = void 0;
const chalk_1 = __importDefault(require("chalk"));
const validationSchemas_1 = require("./validationSchemas");
// TODO temporary escape hatch for alpha-60: to be removed soon
// Our validation schemas might be buggy at first
// will permit users to bypass validation until we fix all validation errors
// see for example: https://github.com/facebook/docusaurus/pull/3120
// Undocumented on purpose, as we don't want users to keep using it over time
// Maybe we'll make this escape hatch official some day, with a better api?
exports.isValidationDisabledEscapeHatch = process.env.DISABLE_DOCUSAURUS_VALIDATION === 'true';
if (exports.isValidationDisabledEscapeHatch) {
    console.error(chalk_1.default.red('You should avoid using DISABLE_DOCUSAURUS_VALIDATION escape hatch, this will be removed'));
}
exports.logValidationBugReportHint = () => {
    console.log(`\n${chalk_1.default.red('A validation error occured.')}${chalk_1.default.cyanBright('\nThe validation system was added recently to Docusaurus as an attempt to avoid user configuration errors.' +
        '\nWe may have made some mistakes.' +
        '\nIf you think your configuration is valid and should keep working, please open a bug report.')}\n`);
};
function normalizePluginOptions(schema, options) {
    // All plugins can be provided an "id" option (multi-instance support)
    // we add schema validation automatically
    const finalSchema = schema.append({
        id: validationSchemas_1.PluginIdSchema,
    });
    const { error, value } = finalSchema.validate(options, {
        convert: false,
    });
    if (error) {
        exports.logValidationBugReportHint();
        if (exports.isValidationDisabledEscapeHatch) {
            console.error(error);
            return options;
        }
        else {
            throw error;
        }
    }
    return value;
}
exports.normalizePluginOptions = normalizePluginOptions;
function normalizeThemeConfig(schema, themeConfig) {
    // A theme should only validate his "slice" of the full themeConfig,
    // not the whole object, so we allow unknown attributes
    // otherwise one theme would fail validating the data of another theme
    const finalSchema = schema.unknown();
    const { error, value } = finalSchema.validate(themeConfig, {
        convert: false,
    });
    if (error) {
        exports.logValidationBugReportHint();
        if (exports.isValidationDisabledEscapeHatch) {
            console.error(error);
            return themeConfig;
        }
        else {
            throw error;
        }
    }
    return value;
}
exports.normalizeThemeConfig = normalizeThemeConfig;
