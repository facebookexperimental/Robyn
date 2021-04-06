/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/// <reference types="react" />
export declare type TranslateParam = {
    message: string;
    id?: string;
    description?: string;
};
export declare function translate({ message, id }: TranslateParam): string;
export declare type TranslateProps = {
    children: string;
    id?: string;
    description?: string;
};
export default function Translate({ children, id }: TranslateProps): JSX.Element;
//# sourceMappingURL=Translate.d.ts.map