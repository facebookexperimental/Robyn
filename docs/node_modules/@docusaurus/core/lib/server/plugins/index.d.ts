/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { LoadContext, PluginConfig, RouteConfig, TranslationFiles, ThemeConfig } from '@docusaurus/types';
import { InitPlugin } from './init';
export declare function sortConfig(routeConfigs: RouteConfig[]): void;
export declare type AllPluginsTranslationFiles = Record<string, // plugin name
Record<string, // plugin id
TranslationFiles>>;
export declare function loadPlugins({ pluginConfigs, context, }: {
    pluginConfigs: PluginConfig[];
    context: LoadContext;
}): Promise<{
    plugins: InitPlugin[];
    pluginsRouteConfigs: RouteConfig[];
    globalData: any;
    themeConfigTranslated: ThemeConfig;
}>;
//# sourceMappingURL=index.d.ts.map