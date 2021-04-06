import { InitPlugin } from '../plugins/init';
import { TranslationFileContent, TranslationFile } from '@docusaurus/types';
export declare type WriteTranslationsOptions = {
    override?: boolean;
    messagePrefix?: string;
};
declare type TranslationContext = {
    siteDir: string;
    locale: string;
};
export declare function ensureTranslationFileContent(content: unknown): asserts content is TranslationFileContent;
export declare function readTranslationFileContent(filePath: string): Promise<TranslationFileContent | undefined>;
export declare function writeTranslationFileContent({ filePath, content: newContent, options, }: {
    filePath: string;
    content: TranslationFileContent;
    options?: WriteTranslationsOptions;
}): Promise<void>;
export declare function getTranslationsDirPath(context: TranslationContext): string;
export declare function getTranslationsLocaleDirPath(context: TranslationContext): string;
export declare function getCodeTranslationsFilePath(context: TranslationContext): string;
export declare function readCodeTranslationFileContent(context: TranslationContext): Promise<TranslationFileContent | undefined>;
export declare function writeCodeTranslations(context: TranslationContext, content: TranslationFileContent, options: WriteTranslationsOptions): Promise<void>;
export declare function writePluginTranslations({ siteDir, plugin, locale, translationFile, options, }: TranslationContext & {
    plugin: InitPlugin;
    translationFile: TranslationFile;
    options?: WriteTranslationsOptions;
}): Promise<void>;
export declare function localizePluginTranslationFile({ siteDir, plugin, locale, translationFile, }: TranslationContext & {
    plugin: InitPlugin;
    translationFile: TranslationFile;
}): Promise<TranslationFile>;
export {};
//# sourceMappingURL=translations.d.ts.map