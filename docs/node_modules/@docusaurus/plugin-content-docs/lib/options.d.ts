/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import * as Joi from 'joi';
import { PluginOptions } from './types';
import { OptionValidationContext, ValidationResult } from '@docusaurus/types';
import { ValidationError } from 'joi';
export declare const DEFAULT_OPTIONS: Omit<PluginOptions, 'id'>;
export declare const OptionsSchema: Joi.ObjectSchema<any>;
export declare function validateOptions({ validate, options, }: OptionValidationContext<PluginOptions, ValidationError>): ValidationResult<PluginOptions, ValidationError>;
//# sourceMappingURL=options.d.ts.map