import { search } from './search';
export function getAlgoliaResults(_ref) {
  var searchClient = _ref.searchClient,
      queries = _ref.queries,
      userAgents = _ref.userAgents;
  return search({
    searchClient: searchClient,
    queries: queries,
    userAgents: userAgents
  }).then(function (response) {
    return response.results;
  });
}