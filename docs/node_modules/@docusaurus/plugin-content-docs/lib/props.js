"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.toVersionMetadataProp = exports.toSidebarsProp = void 0;
const lodash_1 = require("lodash");
function toSidebarsProp(loadedVersion) {
    const docsById = lodash_1.keyBy(loadedVersion.docs, (doc) => doc.id);
    const convertDocLink = (item) => {
        const docId = item.id;
        const docMetadata = docsById[docId];
        if (!docMetadata) {
            throw new Error(`Bad sidebars file. The document id '${docId}' was used in the sidebar, but no document with this id could be found.
Available document ids=
- ${Object.keys(docsById).sort().join('\n- ')}`);
        }
        const { title, permalink, sidebar_label } = docMetadata;
        return {
            type: 'link',
            label: sidebar_label || title,
            href: permalink,
            customProps: item.customProps,
        };
    };
    const normalizeItem = (item) => {
        switch (item.type) {
            case 'category':
                return Object.assign(Object.assign({}, item), { items: item.items.map(normalizeItem) });
            case 'ref':
            case 'doc':
                return convertDocLink(item);
            case 'link':
            default:
                return item;
        }
    };
    // Transform the sidebar so that all sidebar item will be in the
    // form of 'link' or 'category' only.
    // This is what will be passed as props to the UI component.
    return lodash_1.mapValues(loadedVersion.sidebars, (items) => items.map(normalizeItem));
}
exports.toSidebarsProp = toSidebarsProp;
function toVersionMetadataProp(pluginId, loadedVersion) {
    return {
        pluginId,
        version: loadedVersion.versionName,
        label: loadedVersion.versionLabel,
        isLast: loadedVersion.isLast,
        docsSidebars: toSidebarsProp(loadedVersion),
        permalinkToSidebar: loadedVersion.permalinkToSidebar,
    };
}
exports.toVersionMetadataProp = toVersionMetadataProp;
