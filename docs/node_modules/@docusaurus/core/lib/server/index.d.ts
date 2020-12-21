/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { LoadContext, PluginConfig, Props } from '@docusaurus/types';
declare type LoadContextOptions = {
    customOutDir?: string;
    locale?: string;
    localizePath?: boolean;
};
export declare function loadContext(siteDir: string, options?: LoadContextOptions): Promise<LoadContext>;
export declare function loadPluginConfigs(context: LoadContext): PluginConfig[];
export declare function load(siteDir: string, options?: LoadContextOptions): Promise<Props>;
export {};
//# sourceMappingURL=index.d.ts.map