/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
declare type BaseUrlOptions = Partial<{
    forcePrependBaseUrl: boolean;
    absolute: boolean;
}>;
export declare type BaseUrlUtils = {
    withBaseUrl: (url: string, options?: BaseUrlOptions) => string;
};
export declare function useBaseUrlUtils(): BaseUrlUtils;
export default function useBaseUrl(url: string, options?: BaseUrlOptions): string;
export {};
//# sourceMappingURL=useBaseUrl.d.ts.map