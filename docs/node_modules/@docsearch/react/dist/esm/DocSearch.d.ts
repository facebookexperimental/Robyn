import { AutocompleteState, PublicAutocompleteOptions } from '@francoischalifour/autocomplete-core';
import React from 'react';
import { DocSearchHit, InternalDocSearchHit, StoredDocSearchHit, SearchClient } from './types';
export interface DocSearchProps extends Pick<PublicAutocompleteOptions<InternalDocSearchHit>, 'navigator'> {
    appId?: string;
    apiKey: string;
    indexName: string;
    placeholder?: string;
    searchParameters?: any;
    transformItems?(items: DocSearchHit[]): DocSearchHit[];
    hitComponent?(props: {
        hit: InternalDocSearchHit | StoredDocSearchHit;
        children: React.ReactNode;
    }): JSX.Element;
    resultsFooterComponent?(props: {
        state: AutocompleteState<InternalDocSearchHit>;
    }): JSX.Element | null;
    transformSearchClient?(searchClient: SearchClient): SearchClient;
    disableUserPersonalization?: boolean;
    initialQuery?: string;
}
export declare function DocSearch(props: DocSearchProps): JSX.Element;
