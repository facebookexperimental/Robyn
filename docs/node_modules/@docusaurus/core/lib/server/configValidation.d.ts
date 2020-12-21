/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { DocusaurusConfig, I18nConfig } from '@docusaurus/types';
export declare const DEFAULT_I18N_CONFIG: I18nConfig;
export declare const DEFAULT_CONFIG: Pick<DocusaurusConfig, 'i18n' | 'onBrokenLinks' | 'onBrokenMarkdownLinks' | 'onDuplicateRoutes' | 'plugins' | 'themes' | 'presets' | 'customFields' | 'themeConfig' | 'titleDelimiter' | 'noIndex' | 'baseUrlIssueBanner'>;
export declare function validateConfig(config: Partial<DocusaurusConfig>): DocusaurusConfig;
//# sourceMappingURL=configValidation.d.ts.map