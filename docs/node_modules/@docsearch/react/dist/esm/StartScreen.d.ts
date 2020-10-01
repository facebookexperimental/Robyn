/// <reference types="react" />
import { ScreenStateProps } from './ScreenState';
import { InternalDocSearchHit } from './types';
interface StartScreenProps extends ScreenStateProps<InternalDocSearchHit> {
    hasSuggestions: boolean;
}
export declare function StartScreen(props: StartScreenProps): JSX.Element | null;
export {};
