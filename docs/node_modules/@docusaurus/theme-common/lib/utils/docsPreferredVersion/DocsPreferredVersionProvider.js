"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.useDocsPreferredVersionContext = exports.DocsPreferredVersionContextProvider = void 0;
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const react_1 = __importStar(require("react"));
const useThemeConfig_1 = require("../useThemeConfig");
const docsUtils_1 = require("../docsUtils");
const useDocs_1 = require("@theme/hooks/useDocs");
const DocsPreferredVersionStorage_1 = __importDefault(require("./DocsPreferredVersionStorage"));
// Initial state is always null as we can't read localstorage from node SSR
function getInitialState(pluginIds) {
    const initialState = {};
    pluginIds.forEach((pluginId) => {
        initialState[pluginId] = {
            preferredVersionName: null,
        };
    });
    return initialState;
}
// Read storage for all docs plugins
// Assign to each doc plugin a preferred version (if found)
function readStorageState({ pluginIds, versionPersistence, allDocsData, }) {
    // The storage value we read might be stale,
    // and belong to a version that does not exist in the site anymore
    // In such case, we remove the storage value to avoid downstream errors
    function restorePluginState(pluginId) {
        const preferredVersionNameUnsafe = DocsPreferredVersionStorage_1.default.read(pluginId, versionPersistence);
        const pluginData = allDocsData[pluginId];
        const versionExists = pluginData.versions.some((version) => version.name === preferredVersionNameUnsafe);
        if (versionExists) {
            return { preferredVersionName: preferredVersionNameUnsafe };
        }
        else {
            DocsPreferredVersionStorage_1.default.clear(pluginId, versionPersistence);
            return { preferredVersionName: null };
        }
    }
    const initialState = {};
    pluginIds.forEach((pluginId) => {
        initialState[pluginId] = restorePluginState(pluginId);
    });
    return initialState;
}
function useVersionPersistence() {
    return useThemeConfig_1.useThemeConfig().docs.versionPersistence;
}
// Value that  will be accessible through context: [state,api]
function useContextValue() {
    const allDocsData = useDocs_1.useAllDocsData();
    const versionPersistence = useVersionPersistence();
    const pluginIds = react_1.useMemo(() => Object.keys(allDocsData), [allDocsData]);
    // Initial state is empty, as  we can't read browser storage in node/SSR
    const [state, setState] = react_1.useState(() => getInitialState(pluginIds));
    // On mount, we set the state read from browser storage
    react_1.useEffect(() => {
        setState(readStorageState({ allDocsData, versionPersistence, pluginIds }));
    }, [allDocsData, versionPersistence, pluginIds]);
    // The API that we expose to consumer hooks (memo for constant object)
    const api = react_1.useMemo(() => {
        function savePreferredVersion(pluginId, versionName) {
            DocsPreferredVersionStorage_1.default.save(pluginId, versionPersistence, versionName);
            setState((s) => (Object.assign(Object.assign({}, s), { [pluginId]: { preferredVersionName: versionName } })));
        }
        return {
            savePreferredVersion,
        };
    }, [setState]);
    return [state, api];
}
const Context = react_1.createContext(null);
function DocsPreferredVersionContextProvider({ children, }) {
    if (docsUtils_1.isDocsPluginEnabled) {
        return (react_1.default.createElement(DocsPreferredVersionContextProviderUnsafe, null, children));
    }
    else {
        return react_1.default.createElement(react_1.default.Fragment, null, children);
    }
}
exports.DocsPreferredVersionContextProvider = DocsPreferredVersionContextProvider;
function DocsPreferredVersionContextProviderUnsafe({ children, }) {
    const contextValue = useContextValue();
    return react_1.default.createElement(Context.Provider, { value: contextValue }, children);
}
function useDocsPreferredVersionContext() {
    const value = react_1.useContext(Context);
    if (!value) {
        throw new Error("Can't find docs preferred context, maybe you forgot to use the DocsPreferredVersionContextProvider ?");
    }
    return value;
}
exports.useDocsPreferredVersionContext = useDocsPreferredVersionContext;
