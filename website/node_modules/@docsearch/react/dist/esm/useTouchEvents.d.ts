import { AutocompleteApi } from '@francoischalifour/autocomplete-core';
interface UseTouchEventsProps {
    getEnvironmentProps: AutocompleteApi<any>['getEnvironmentProps'];
    dropdownElement: HTMLDivElement | null;
    searchBoxElement: HTMLDivElement | null;
    inputElement: HTMLInputElement | null;
}
export declare function useTouchEvents({ getEnvironmentProps, dropdownElement, searchBoxElement, inputElement, }: UseTouchEventsProps): void;
export {};
