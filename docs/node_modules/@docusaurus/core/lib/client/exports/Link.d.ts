/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { ReactNode } from 'react';
declare global {
    interface Window {
        docusaurus: any;
    }
}
interface Props {
    readonly isNavLink?: boolean;
    readonly to?: string;
    readonly href?: string;
    readonly activeClassName?: string;
    readonly children?: ReactNode;
    readonly isActive?: () => boolean;
    readonly autoAddBaseUrl?: boolean;
    readonly 'data-noBrokenLinkCheck'?: boolean;
}
declare function Link({ isNavLink, to, href, activeClassName, isActive, 'data-noBrokenLinkCheck': noBrokenLinkCheck, autoAddBaseUrl, ...props }: Props): JSX.Element;
export default Link;
//# sourceMappingURL=Link.d.ts.map