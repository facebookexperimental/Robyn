/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { LoadContext, Plugin, PluginOptions, PluginConfig, DocusaurusPluginVersionInformation } from '@docusaurus/types';
export declare type InitPlugin = Plugin<unknown> & {
    readonly options: PluginOptions;
    readonly version: DocusaurusPluginVersionInformation;
};
export default function initPlugins({ pluginConfigs, context, }: {
    pluginConfigs: PluginConfig[];
    context: LoadContext;
}): InitPlugin[];
//# sourceMappingURL=init.d.ts.map