import { TransformOptions } from '@babel/core';
import { TranslationFileContent, TranslationMessage } from '@docusaurus/types';
import { InitPlugin } from '../plugins/init';
export declare function extractPluginsSourceCodeTranslations(plugins: InitPlugin[], babelOptions: TransformOptions): Promise<TranslationFileContent>;
declare type SourceCodeFileTranslations = {
    sourceCodeFilePath: string;
    translations: Record<string, TranslationMessage>;
    warnings: string[];
};
export declare function extractSourceCodeFileTranslations(sourceCodeFilePath: string, babelOptions: TransformOptions): Promise<SourceCodeFileTranslations>;
export {};
//# sourceMappingURL=translationsExtractor.d.ts.map