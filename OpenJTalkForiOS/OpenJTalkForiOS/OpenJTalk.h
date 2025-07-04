//
//  OpenJTalk.h
//  SpeechRecognizerTest
//
//  Created by Yasuo Hasegawa on 2025/06/26.
//

#ifndef OpenJTalk_h
#define OpenJTalk_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface OpenJTalk : NSObject

- (void)synthesize:(NSString *)text pitch:(double)pitch toURL:(NSURL *)outputURL completion:(void (^)(NSError * _Nullable error))completion;

@end

NS_ASSUME_NONNULL_END

#endif /* OpenJTalk_h */
