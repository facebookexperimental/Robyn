import { SearchForFacetValuesQueryParams, SearchOptions } from '@algolia/client-search';
import { SearchClient } from 'algoliasearch/lite';
import { UserAgent } from './UserAgent';
declare type FacetQuery = {
    indexName: string;
    params: SearchForFacetValuesQueryParams & SearchOptions;
};
export interface SearchForFacetValuesParams {
    searchClient: SearchClient;
    queries: FacetQuery[];
    userAgents?: UserAgent[];
}
export declare function searchForFacetValues({ searchClient, queries, userAgents, }: SearchForFacetValuesParams): Readonly<Promise<readonly import("@algolia/client-search").SearchForFacetValuesResponse[]>>;
export {};
