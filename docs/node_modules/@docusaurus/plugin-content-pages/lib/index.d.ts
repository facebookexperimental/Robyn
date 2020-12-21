/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { LoadContext, Plugin, OptionValidationContext, ValidationResult } from '@docusaurus/types';
import { PluginOptionSchema } from './pluginOptionSchema';
import { ValidationError } from 'joi';
import { PluginOptions, LoadedContent, PagesContentPaths } from './types';
export declare function getContentPathList(contentPaths: PagesContentPaths): string[];
export default function pluginContentPages(context: LoadContext, options: PluginOptions): Plugin<LoadedContent | null, typeof PluginOptionSchema>;
export declare function validateOptions({ validate, options, }: OptionValidationContext<PluginOptions, ValidationError>): ValidationResult<PluginOptions, ValidationError>;
//# sourceMappingURL=index.d.ts.map