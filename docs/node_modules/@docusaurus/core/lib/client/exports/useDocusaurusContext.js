/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { useContext } from 'react';
import context from './context';
function useDocusaurusContext() {
    const docusaurusContext = useContext(context);
    if (docusaurusContext === null) {
        // should not happen normally
        throw new Error('Docusaurus context not provided');
    }
    return docusaurusContext;
}
export default useDocusaurusContext;
