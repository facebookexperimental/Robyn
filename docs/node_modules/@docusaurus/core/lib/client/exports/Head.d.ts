/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { ReactNode } from 'react';
import { HelmetProps } from 'react-helmet';
declare type HeadProps = HelmetProps & {
    children: ReactNode;
};
declare function Head(props: HeadProps): JSX.Element;
export default Head;
//# sourceMappingURL=Head.d.ts.map