# reproduce-whisper-expected-outputs
This repo contains reproducers for [Transformers'](https://github.com/huggingface/transformers) [Whisper integration tests](https://github.com/huggingface/transformers/blob/37ea04013b34b39c01b51aeaacd8d56f2c62a7eb/tests/models/whisper/test_modeling_whisper.py#L4). 

To compute the correct excepted outputs, it is necessary to work from a very simple fork of the original OpenAI Whisper implementation. Indeed, the extraction of the mel spectrogram in WhisperFeatureExtractor diverges slightly from OpenAI's one: we pad the audio array to 30sec/ to longest with 0.0s and then extract our spectrogram through batched STFT while OpenAI's one will add 30sec of 0.0s to the audio array (and not pad to 30sec). This way, the are sure that model inputs for our and OpenAI's implementations are exactly the same.

With this, we can use the following protocol to compute the expected outputs for the tests:

extract mel inputs using the test's implementation (so using the WhisperFeatureExtractor)
infer OpenAI's model through the above explained fork directly from the mel input
