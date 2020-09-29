import { AutocompleteApi, AutocompleteState } from '@francoischalifour/autocomplete-core';
import React from 'react';
import { DocSearchProps } from './DocSearch';
import { StoredSearchPlugin } from './stored-searches';
import { InternalDocSearchHit, StoredDocSearchHit } from './types';
export interface ScreenStateProps<TItem> extends AutocompleteApi<TItem, React.FormEvent, React.MouseEvent, React.KeyboardEvent> {
    state: AutocompleteState<TItem>;
    recentSearches: StoredSearchPlugin<StoredDocSearchHit>;
    favoriteSearches: StoredSearchPlugin<StoredDocSearchHit>;
    onItemClick(item: InternalDocSearchHit): void;
    inputRef: React.MutableRefObject<null | HTMLInputElement>;
    hitComponent: DocSearchProps['hitComponent'];
    indexName: DocSearchProps['indexName'];
    disableUserPersonalization: boolean;
    resultsFooterComponent: DocSearchProps['resultsFooterComponent'];
}
export declare const ScreenState: React.MemoExoticComponent<(props: ScreenStateProps<InternalDocSearchHit>) => JSX.Element>;
