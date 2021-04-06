import { Hit } from '@algolia/client-search';
import { SearchParams } from './search';
export declare function getAlgoliaHits<TRecord>({ searchClient, queries, userAgents, }: SearchParams): Promise<Array<Array<Hit<TRecord>>>>;
