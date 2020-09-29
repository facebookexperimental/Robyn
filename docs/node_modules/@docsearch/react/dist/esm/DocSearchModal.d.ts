/// <reference types="react" />
import { DocSearchProps } from './DocSearch';
export interface DocSearchModalProps extends DocSearchProps {
    initialScrollY: number;
    onClose?(): void;
}
export declare function DocSearchModal({ appId, apiKey, indexName, placeholder, searchParameters, onClose, transformItems, hitComponent, resultsFooterComponent, navigator, initialScrollY, transformSearchClient, disableUserPersonalization, initialQuery: initialQueryFromProp, }: DocSearchModalProps): JSX.Element;
