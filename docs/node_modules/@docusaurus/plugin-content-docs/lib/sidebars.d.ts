/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { Sidebars, SidebarItem, SidebarItemLink, SidebarItemDoc, Sidebar, SidebarItemCategory } from './types';
export declare function loadSidebars(sidebarFilePath: string): Sidebars;
export declare function collectSidebarDocItems(sidebar: Sidebar): SidebarItemDoc[];
export declare function collectSidebarCategories(sidebar: Sidebar): SidebarItemCategory[];
export declare function collectSidebarLinks(sidebar: Sidebar): SidebarItemLink[];
export declare function transformSidebarItems(sidebar: Sidebar, updateFn: (item: SidebarItem) => SidebarItem): Sidebar;
export declare function collectSidebarsDocIds(sidebars: Sidebars): Record<string, string[]>;
export declare function createSidebarsUtils(sidebars: Sidebars): Record<string, Function>;
//# sourceMappingURL=sidebars.d.ts.map