import { searchForFacetValues } from './searchForFacetValues';
export function getAlgoliaFacetHits(_ref) {
  var searchClient = _ref.searchClient,
      queries = _ref.queries,
      userAgents = _ref.userAgents;
  return searchForFacetValues({
    searchClient: searchClient,
    queries: queries,
    userAgents: userAgents
  }).then(function (response) {
    return response.map(function (result) {
      return result.facetHits.map(function (facetHit) {
        return {
          label: facetHit.value,
          count: facetHit.count,
          _highlightResult: {
            label: {
              value: facetHit.highlighted
            }
          }
        };
      });
    });
  });
}