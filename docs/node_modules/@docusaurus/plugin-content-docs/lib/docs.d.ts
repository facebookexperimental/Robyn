/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { LoadContext } from '@docusaurus/types';
import { DocFile, DocMetadataBase, MetadataOptions, PluginOptions, VersionMetadata } from './types';
declare type LastUpdateOptions = Pick<PluginOptions, 'showLastUpdateAuthor' | 'showLastUpdateTime'>;
export declare function readDocFile(versionMetadata: Pick<VersionMetadata, 'docsDirPath' | 'docsDirPathLocalized'>, source: string, options: LastUpdateOptions): Promise<DocFile>;
export declare function readVersionDocs(versionMetadata: VersionMetadata, options: Pick<PluginOptions, 'include' | 'showLastUpdateAuthor' | 'showLastUpdateTime'>): Promise<DocFile[]>;
export declare function processDocMetadata({ docFile, versionMetadata, context, options, }: {
    docFile: DocFile;
    versionMetadata: VersionMetadata;
    context: LoadContext;
    options: MetadataOptions;
}): DocMetadataBase;
export {};
//# sourceMappingURL=docs.d.ts.map