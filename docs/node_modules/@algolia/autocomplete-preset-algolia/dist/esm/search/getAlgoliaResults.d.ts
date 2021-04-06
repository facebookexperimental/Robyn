import { SearchResponse } from '@algolia/client-search';
import { SearchParams } from './search';
export declare function getAlgoliaResults<TRecord>({ searchClient, queries, userAgents, }: SearchParams): Promise<Array<SearchResponse<TRecord>>>;
