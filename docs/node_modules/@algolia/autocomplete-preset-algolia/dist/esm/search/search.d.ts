import { MultipleQueriesQuery } from '@algolia/client-search';
import { SearchClient } from 'algoliasearch/lite';
import { UserAgent } from './UserAgent';
export interface SearchParams {
    searchClient: SearchClient;
    queries: MultipleQueriesQuery[];
    userAgents?: UserAgent[];
}
export declare function search<TRecord>({ searchClient, queries, userAgents, }: SearchParams): Readonly<Promise<import("@algolia/client-search").MultipleQueriesResponse<TRecord>>>;
