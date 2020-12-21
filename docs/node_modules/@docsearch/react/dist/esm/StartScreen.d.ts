/// <reference types="react" />
import { ScreenStateProps } from './ScreenState';
import { InternalDocSearchHit } from './types';
interface StartScreenProps extends ScreenStateProps<InternalDocSearchHit> {
    hasCollections: boolean;
}
export declare function StartScreen(props: StartScreenProps): JSX.Element | null;
export {};
