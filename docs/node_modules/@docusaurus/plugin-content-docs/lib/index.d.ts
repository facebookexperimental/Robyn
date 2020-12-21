/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { LoadContext, Plugin } from '@docusaurus/types';
import { PluginOptions, LoadedContent } from './types';
import { OptionsSchema } from './options';
export default function pluginContentDocs(context: LoadContext, options: PluginOptions): Plugin<LoadedContent, typeof OptionsSchema>;
export { validateOptions } from './options';
//# sourceMappingURL=index.d.ts.map