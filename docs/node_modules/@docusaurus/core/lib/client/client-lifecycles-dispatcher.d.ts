/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
interface Dispatchers {
    onRouteUpdate: (...args: any) => void;
    onRouteUpdateDelayed: (...args: any) => void;
}
declare const clientLifecyclesDispatchers: Dispatchers;
export default clientLifecyclesDispatchers;
//# sourceMappingURL=client-lifecycles-dispatcher.d.ts.map