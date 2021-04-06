/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import * as Joi from 'joi';
export declare const isValidationDisabledEscapeHatch: boolean;
export declare const logValidationBugReportHint: () => void;
export declare function normalizePluginOptions<T extends {
    id?: string;
}>(schema: Joi.ObjectSchema<T>, options: unknown): any;
export declare function normalizeThemeConfig<T>(schema: Joi.ObjectSchema<T>, themeConfig: unknown): any;
//# sourceMappingURL=validationUtils.d.ts.map