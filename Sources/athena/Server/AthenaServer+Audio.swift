import AthenaCore
import AthenaDeploy
import AthenaEmbedding
import AthenaLLM
import AthenaServerKit
import AthenaStore
import AthenaStructured
import AthenaTranscription
import Foundation
import HTTPTypes
import Hummingbird
import MLX
import HummingbirdCore
import HummingbirdTLS
import Logging
import NIOCore
import NIOSSL

// The media surface: `/v1/audio/*` (transcription, diarization, speaker
// embeddings) and the ADR 022 `/v1/video/transcriptions` orchestration. Each
// funnels through the shared `extractUploadFile` multipart preamble (WP9) and
// the governed transcription/diarization/speaker tenants.
extension AthenaServer {
    /// Register the media routes (`AthenaServer+Audio.swift`). Called from `run()`.
    func registerAudioRoutes(_ router: Router<AppRequestContext>) {
        router.post("/v1/audio/transcriptions") { request, _ -> Response in
            await self.handleTranscriptions(request)
        }

        router.post("/v1/audio/diarizations") { request, _ -> Response in
            await self.handleDiarizations(request)
        }

        router.post("/v1/audio/embeddings") { request, _ -> Response in
            await self.handleSpeakerEmbeddings(request)
        }

        // ADR 022 — Athena-native (NOT OpenAI; OpenAI has no video API). Demux
        // the audio track and transcribe it via the same Whisper/Parakeet
        // tenant; the response shape mirrors /v1/audio/transcriptions.
        router.post("/v1/video/transcriptions") { request, _ -> Response in
            await self.handleVideoTranscriptions(request)
        }
    }

    /// WP9 — the shared multipart upload preamble for the media routes
    /// (transcription, video, diarization, speaker-embedding): content-type +
    /// boundary → ADR-017 cap fast-fail (declared `Content-Length`) →
    /// `collect(upTo:cap)` with a streamed 413 backstop → parse → the required
    /// non-empty `file` part. `cap` is the modality's byte ceiling (audio vs
    /// video), so one helper closes the drift between the four hand-copied
    /// blocks (the audio/video cap had already diverged).
    /// Test coverage (2026-07-02 audit WP9, decided): the cap/413 decision
    /// algebra is unit-pinned in `AthenaServerKit/UploadLimit` (ADR 008); this
    /// helper is HTTP plumbing in the executable target (un-importable under
    /// `swift test`) and is pinned by the four routes' e2e coverage instead.
    private func extractUploadFile(
        _ request: Request, cap: Int
    ) async -> Outcome<(form: MultipartForm, file: MultipartForm.Part)> {
        guard
            let ct = request.headers[.contentType],
            let boundary = MultipartForm.boundary(fromContentType: ct)
        else {
            return .fail(
                Self.error(
                    status: .badRequest,
                    message: "expected multipart/form-data with a boundary",
                    type: "invalid_request_error", code: "invalid_content_type"))
        }
        if let tooLarge = Self.payloadTooLarge(request, cap: cap) {
            return .fail(tooLarge)
        }
        let body: Data
        do {
            let buffer = try await request.body.collect(upTo: cap)
            body = Data(buffer: buffer)
        } catch is NIOTooManyBytesError {
            return .fail(Self.tooLargeResponse(cap: cap))
        } catch {
            return .fail(
                Self.error(
                    status: .badRequest, message: "invalid request body",
                    type: "invalid_request_error", code: "invalid_body"))
        }
        guard
            let form = MultipartForm(body: body, boundary: boundary),
            let file = form.first("file"), !file.data.isEmpty
        else {
            return .fail(
                Self.error(
                    status: .badRequest,
                    message: "missing required 'file' part",
                    type: "invalid_request_error", code: "missing_file"))
        }
        return .ok((form, file))
    }

    private func handleTranscriptions(_ request: Request) async -> Response
    {
        let t0 = Date()
        let upload = await extractUploadFile(request, cap: maxAudioUploadBytes)
        guard case .ok(let (form, file)) = upload else { return upload.orFail }

        do {
            // M43.2: non-blocking cold-load (see /v1/embeddings).
            switch try await governor.awaitLoad(.transcription, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.transcription)
            }
            // M41.3: a `model` form field selects among the operator-
            // declared whisper allowlist; an unknown id ⇒ 400
            // model_not_available via the classified path. M41.4: an
            // actual rebind is audited (`model.rebind` trigger=inference).
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .transcription, target: m)
            }
        } catch {
            // issue #6: classify (see textEmbedding handler) — no leak, right code.
            return Self.classified(error, module: .transcription)
        }

        // Word timestamps are an opt-in of verbose_json only
        // (`timestamp_granularities[]=word`); every other format is
        // byte-unchanged and never triggers the alignment pass.
        let wantWords =
            form.text("response_format") == "verbose_json"
            && form.texts("timestamp_granularities[]").contains("word")

        let result: TranscriptionResult
        do {
            result = try await transcription.transcribe(
                audio: file.data, filename: file.filename,
                language: form.text("language"),
                wordTimestamps: wantWords,
                requestedModel: form.text("model"))
        } catch {
            return Self.classified(error, module: .transcription)
        }
        // M56 — per-request summary (the format serialization below is
        // cheap; this captures the transcription work).
        Logger(label: AthenaLogLabel.model(.transcription)).notice(
            """
            transcription done segments=\(result.segments.count) \
            audio_secs=\(String(format: "%.1f", result.duration)) \
            lang=\(result.language ?? "auto") words=\(wantWords) \
            elapsed_ms=\(Self.elapsedMs(t0))
            """)

        // Resolve diarization turns for the speaker-labeled formats: opt-in
        // `diarize=true` on verbose_json (M4.3c), or diarized_json — which
        // IMPLIES diarization (ADR 013 #3/#4a) and reuses the verbose envelope
        // (`speaker` is Athena's Int id, not OpenAI's string label; the
        // standalone /v1/audio/diarizations route stays canonical). Every
        // other format needs none.
        let fmt = form.text("response_format")
        var turns: [DiarizationTurn] = []
        if (fmt == "verbose_json" && form.text("diarize") == "true")
            || fmt == "diarized_json"
        {
            switch await diarizeTurns(
                audio: file.data, filename: file.filename)
            {
            case .fail(let r): return r
            case .ok(let t): turns = t
            }
        }
        // diarized_json normalizes to the verbose encoder with mandatory turns
        // (empty turns ⇒ speaker nil, same as pre-refactor Self.speaker([])).
        return Self.encodeTranscription(
            result, format: fmt == "diarized_json" ? "verbose_json" : fmt,
            wantWords: wantWords, turns: turns)
    }

    /// Serialize a `TranscriptionResult` into the requested `response_format`
    /// (text/srt/vtt/verbose_json/json) — shared by `/v1/audio/transcriptions`
    /// and `/v1/video/transcriptions` (video passes `turns: []`; audio's
    /// diarized_json normalizes to verbose_json with mandatory `turns`).
    /// `turns` empty ⇒ no speaker labels; per-segment/word timings are only
    /// present when `wantWords` drove the transcribe pass.
    static func encodeTranscription(
        _ result: TranscriptionResult, format: String?, wantWords: Bool,
        turns: [DiarizationTurn]
    ) -> Response {
        func plain(_ s: String, _ type: String) -> Response {
            var headers = HTTPFields()
            headers[.contentType] = type
            var buf = ByteBuffer()
            buf.writeString(s)
            return Response(
                status: .ok, headers: headers,
                body: ResponseBody(byteBuffer: buf))
        }
        func words(_ ws: [WordTiming]?) -> [WordTimestamp]? {
            ws.map {
                $0.map {
                    WordTimestamp(
                        word: $0.word, start: $0.start, end: $0.end,
                        probability: $0.probability)
                }
            }
        }
        switch format {
        case "text":
            return plain(result.text, "text/plain; charset=utf-8")
        case "srt":
            return plain(
                TranscriptionFormat.srt(result.segments),
                "text/plain; charset=utf-8")
        case "vtt":
            return plain(
                TranscriptionFormat.vtt(result.segments),
                "text/vtt; charset=utf-8")
        case "verbose_json":
            return Self.json(
                VerboseTranscriptionResponse(
                    task: "transcribe", language: result.language,
                    duration: result.duration, text: result.text,
                    segments: result.segments.enumerated().map {
                        VerboseSegment(
                            id: $0.offset, start: $0.element.start,
                            end: $0.element.end, text: $0.element.text,
                            avg_logprob: $0.element.avgLogprob,
                            speaker: turns.isEmpty
                                ? nil
                                : Self.speaker(
                                    start: $0.element.start,
                                    end: $0.element.end, turns: turns),
                            words: words($0.element.words))
                    },
                    words: wantWords ? words(result.words) : nil))
        default:  // "json" / nil
            return Self.json(TranscriptionResponse(text: result.text))
        }
    }

    /// Run end-to-end (Sortformer) diarization to tag transcription segments
    /// with speaker turns — shared by `verbose_json` (opt-in `diarize=true`)
    /// and `diarized_json` (implicit). The diarization slot is a single global
    /// tenant (ADR 011); if a pyannote *segmentation* model is resident (e.g. a
    /// prior `method=pyannote` request rebound it), this can't run and returns a
    /// clear 409 rather than a misleading message (the canonical diarization
    /// surface is the standalone /v1/audio/diarizations route, ADR 013).
    private func diarizeTurns(
        audio: Data, filename: String?
    ) async -> Outcome<[DiarizationTurn]> {
        do {
            // M43.2: non-blocking cold-load.
            switch try await governor.awaitLoad(
                .diarization, within: coldLoadWaitSecs)
            {
            case .loaded: break
            case .loading: return .fail(Self.coldLoadResponse(.diarization))
            }
            guard await diarization.residentBackend() == .sortformer else {
                return .fail(
                    Self.error(
                        status: .conflict,
                        message: "diarization needs an end-to-end "
                            + "(Sortformer) model, but a segmentation model is "
                            + "currently resident in the single diarization "
                            + "slot. Diarize separately via POST "
                            + "/v1/audio/diarizations, or select a Sortformer "
                            + "diarization model.",
                        type: "invalid_request_error",
                        code: "diarization_backend_conflict"))
            }
            return .ok(
                try await diarization.diarize(
                    audio: audio, filename: filename, requestedModel: nil
                ).turns)
        } catch {
            return .fail(Self.classified(error, module: .diarization))
        }
    }

    /// The speaker whose turn most overlaps `[start,end]`, or nil if
    /// none overlap (M4.3c Sortformer↔Whisper alignment).
    private static func speaker(
        start: Double, end: Double, turns: [DiarizationTurn]
    ) -> Int? {
        var best: (speaker: Int, overlap: Double)?
        for t in turns {
            let ov = min(end, t.end) - max(start, t.start)
            if ov > 0, best == nil || ov > best!.overlap {
                best = (t.speaker, ov)
            }
        }
        return best?.speaker
    }

    /// ADR 022 M78.1 — `POST /v1/video/transcriptions` (Athena-native, NOT
    /// OpenAI). Demux the audio track out of the uploaded video and transcribe
    /// it via the same Whisper/Parakeet tenant; the response shapes mirror
    /// `/v1/audio/transcriptions` so an existing transcription consumer reuses
    /// its parser. Bounded by `maxVideoUploadBytes`. (`diarize=true` on video is
    /// not yet wired — a 501; transcribe, then diarize the extracted audio via
    /// `/v1/audio/diarizations`.)
    private func handleVideoTranscriptions(_ request: Request) async -> Response
    {
        let t0 = Date()
        let upload = await extractUploadFile(request, cap: maxVideoUploadBytes)
        guard case .ok(let (form, file)) = upload else { return upload.orFail }

        // diarization on video is a 501 (not yet wired) — fail fast before any
        // load/decode so the caller gets the clear answer immediately. Both the
        // `diarize=true` flag and the `diarized_json` format (which implies
        // diarization, ADR 013 #3) take this path.
        if form.text("diarize") == "true"
            || form.text("response_format") == "diarized_json"
        {
            return Self.classified(
                AthenaError.notImplemented(
                    feature: "diarization on /v1/video/transcriptions "
                        + "(diarize=true or response_format=diarized_json) — "
                        + "transcribe, then POST the extracted audio to "
                        + "/v1/audio/diarizations"),
                module: .transcription)
        }

        do {
            switch try await governor.awaitLoad(
                .transcription, within: coldLoadWaitSecs)
            {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.transcription)
            }
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .transcription, target: m)
            }
        } catch {
            return Self.classified(error, module: .transcription)
        }

        let wantWords =
            form.text("response_format") == "verbose_json"
            && form.texts("timestamp_granularities[]").contains("word")

        // Demux the audio track to PCM straight from the in-memory upload bytes
        // (Option D, ADR 025 S5 — no temp file), then reuse the shared
        // transcribePCM seam (S2). The extracted PCM funnels through the same
        // floor/ceiling — a degenerate video is a 4xx here.
        let result: TranscriptionResult
        do {
            var pcm = try await VideoAudioTrack.extractPCM(
                from: file.data, filename: file.filename, module: .transcription)
            defer { ProcessHardening.secureZero(&pcm) }  // ADR 024 T2
            result = try await transcription.transcribePCM(
                pcm, language: form.text("language"),
                wordTimestamps: wantWords,
                requestedModel: form.text("model"))
        } catch {
            return Self.classified(error, module: .transcription)
        }
        Logger(label: AthenaLogLabel.model(.transcription)).notice(
            """
            video transcription done segments=\(result.segments.count) \
            audio_secs=\(String(format: "%.1f", result.duration)) \
            lang=\(result.language ?? "auto") words=\(wantWords) \
            elapsed_ms=\(Self.elapsedMs(t0))
            """)

        // Video is a strict subset — no diarization (rejected above), so no
        // speaker turns. Same encoder as /v1/audio/transcriptions.
        return Self.encodeTranscription(
            result, format: form.text("response_format"),
            wantWords: wantWords, turns: [])
    }

    private func handleDiarizations(_ request: Request) async -> Response
    {
        let t0 = Date()
        let upload = await extractUploadFile(request, cap: maxAudioUploadBytes)
        guard case .ok(let (form, file)) = upload else { return upload.orFail }

        // Method select (ADR 018). Default `sortformer` (fast, end-to-end,
        // ≤4 speakers); `cluster` = naive-window embedding cluster (M25.3,
        // >4, no overlap); `pyannote` = learned segmentation + embed + global
        // cluster (overlap-aware, arbitrary speakers). `model` selects weights
        // within the chosen method's family.
        switch (form.text("method") ?? "").lowercased() {
        case "", "sortformer":
            break  // fall through to the Sortformer path below
        case "cluster":
            return await handleClusterDiarization(file: file, form: form)
        case "pyannote":
            return await handlePyannoteDiarization(
                request: request, file: file, form: form)
        case let other:
            return Self.classified(
                AthenaError.diarizationMethodInvalid(
                    method: other,
                    reason: "unknown method — use sortformer, cluster, "
                        + "or pyannote"),
                module: .diarization)
        }

        do {
            // M43.2: non-blocking cold-load.
            switch try await governor.awaitLoad(.diarization, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.diarization)
            }
            // M41.3 per-request diarization model selection;
            // M41.4 audited on a real resident-id change.
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .diarization, target: m)
            }
        } catch {
            // issue #6: classify (see textEmbedding handler) — no leak, right code.
            return Self.classified(error, module: .diarization)
        }

        let r: DiarizationResult
        do {
            r = try await diarization.diarize(
                audio: file.data, filename: file.filename,
                requestedModel: form.text("model"))
        } catch {
            return Self.classified(error, module: .diarization)
        }
        // M56 — per-request summary.
        Logger(label: AthenaLogLabel.model(.diarization)).notice(
            """
            diarization done method=sortformer \
            speakers=\(r.numSpeakers) turns=\(r.turns.count) \
            elapsed_ms=\(Self.elapsedMs(t0))
            """)
        return Self.json(
            DiarizationResponse(
                num_speakers: r.numSpeakers,
                segments: r.turns.map {
                    DiarizationSegmentDTO(
                        start: $0.start, end: $0.end,
                        speaker: $0.speaker)
                }))
    }

    /// ADR 018 — pyannote pipeline: learned PyanNet segmentation → WeSpeaker
    /// embed each locally-active region → GLOBAL agglomerative cluster
    /// (same-window cannot-link) → overlap-aware turns with file-stable speaker
    /// ids. The overlap-aware path the naive `cluster` method cannot produce.
    /// The resident diarization model must be a pyannote segmentation
    /// checkpoint (select via `model=`); `num_speakers`/`min_speakers`/
    /// `max_speakers`/`threshold` tune the global clustering.
    private func handlePyannoteDiarization(
        request: Request, file: MultipartForm.Part, form: MultipartForm
    ) async -> Response {
        let t0 = Date()
        let numSpeakers = form.text("num_speakers").flatMap(Int.init)
        let minSpeakers = form.text("min_speakers").flatMap(Int.init)
        let maxSpeakers = form.text("max_speakers").flatMap(Int.init)
        let threshold = form.text("threshold").flatMap(Float.init) ?? 0.75

        do {
            // Cold-load the segmentation slot + optional per-request model
            // selection, same as the Sortformer path (shared 100 MiB cap +
            // cold-load 503 behavior).
            switch try await governor.awaitLoad(
                .diarization, within: coldLoadWaitSecs)
            {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.diarization)
            }
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .diarization, target: m)
            }
            // The pyannote pipeline also needs the speaker-embedding model.
            switch try await governor.awaitLoad(
                .speakerEmbedding, within: coldLoadWaitSecs)
            {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.speakerEmbedding)
            }
        } catch {
            return Self.classified(error, module: .diarization)
        }

        // Method/model match: pyannote requires a segmentation-backed model.
        let backend = await diarization.residentBackend()
        guard backend == .pyannoteSegmentation else {
            return Self.classified(
                AthenaError.diarizationMethodInvalid(
                    method: "pyannote",
                    reason: "the resident diarization model is not a "
                        + "segmentation model — select one with `model=` "
                        + "(operator must pull a pyannote-segmentation model "
                        + "into the diarization allowlist)"),
                module: .diarization)
        }

        // 1. Learned segmentation → per-window locally-tagged regions.
        let regions: [SpeakerActivityRegion]
        do {
            regions = try await diarization.segment(
                audio: file.data, filename: file.filename,
                requestedModel: form.text("model"))
        } catch {
            return Self.classified(error, module: .diarization)
        }
        if regions.isEmpty {
            return Self.json(
                DiarizationResponse(num_speakers: 0, segments: []))
        }

        // 2. Embed each region with WeSpeaker (256-d, stable model — ADR 018).
        let embResult: SpeakerEmbeddingResult
        do {
            embResult = try await speakerEmbedding.embed(
                audio: file.data, filename: file.filename,
                segments: regions.map {
                    SpeakerSegmentRequest(start: $0.start, end: $0.end)
                }, requestedModel: nil)
        } catch {
            return Self.classified(error, module: .speakerEmbedding)
        }
        guard embResult.segments.count == regions.count else {
            return Self.classified(
                AthenaError.moduleLoadFailed(
                    .speakerEmbedding,
                    reason: "embedding count \(embResult.segments.count) "
                        + "≠ region count \(regions.count)"),
                module: .speakerEmbedding)
        }

        // 3. GLOBAL cluster with same-window cannot-link → file-stable ids.
        let embeddings = embResult.segments.map { $0.embedding }
        var labels = AgglomerativeClustering.cluster(
            embeddings,
            numClusters: numSpeakers, threshold: threshold,
            maxClusters: maxSpeakers, minClusters: minSpeakers ?? 1,
            cannotLink: PyannoteSegmentationDecode.sameWindowCannotLink(regions))

        let regionDurations = regions.map { $0.end - $0.start }
        if let target = numSpeakers {
            // Exact count: the agglomerative cut can stick *above* the target
            // because same-window cannot-link forbids the final merges, so
            // force exactly N (override the constraint — the user asked for N).
            labels = PyannoteSegmentationDecode.reduceToTargetClusters(
                embeddings: embeddings, labels: labels,
                durations: regionDurations, target: target)
        } else {
            // Auto mode: dissolve tiny clusters into real speakers (pyannote
            // min_cluster_size) so noisy short/overlap embeddings on long messy
            // audio don't inflate the count.
            let minClusterSeconds =
                form.text("min_cluster_seconds").flatMap(Double.init) ?? 6.0
            labels = PyannoteSegmentationDecode.reassignSmallClusters(
                embeddings: embeddings, labels: labels,
                durations: regionDurations, minDuration: minClusterSeconds)
        }

        // 4. Overlap-aware turns: one turn per region at its global id, then
        //    merge each speaker's overlapping/adjacent turns (cross-speaker
        //    overlap is preserved).
        let rawTurns = zip(regions, labels).map {
            DiarizationTurn(start: $0.start, end: $0.end, speaker: $1)
        }
        let turns = PyannoteSegmentationDecode.mergeSameSpeakerTurns(rawTurns)
        let speakers = Set(labels).count

        Logger(label: AthenaLogLabel.model(.diarization)).notice(
            """
            diarization done method=pyannote \
            speakers=\(speakers) regions=\(regions.count) \
            turns=\(turns.count) elapsed_ms=\(Self.elapsedMs(t0))
            """)
        return Self.json(
            DiarizationResponse(
                num_speakers: speakers,
                segments: turns.map {
                    DiarizationSegmentDTO(
                        start: $0.start, end: $0.end, speaker: $0.speaker)
                }))
    }

    /// M25.3 embedding+clustering diarizer: window → WeSpeaker embed →
    /// agglomerative cluster → merge same-speaker windows into turns.
    /// Recovers >4 speakers, which the offline Sortformer cannot.
    private func handleClusterDiarization(
        file: MultipartForm.Part, form: MultipartForm
    ) async -> Response {
        let t0 = Date()
        let numSpeakers = form.text("num_speakers").flatMap(Int.init)
        let maxSpeakers = form.text("max_speakers").flatMap(Int.init)
        let threshold = form.text("threshold").flatMap(Float.init) ?? 0.75

        do {
            // M43.2: non-blocking cold-load.
            switch try await governor.awaitLoad(.speakerEmbedding, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.speakerEmbedding)
            }
        } catch {
            // issue #6: classify (see textEmbedding handler) — no leak, right code.
            return Self.classified(error, module: .speakerEmbedding)
        }

        let we: SpeakerEmbeddingResult
        do {
            we = try await speakerEmbedding.windowEmbeddings(
                audio: file.data, filename: file.filename,
                windowSeconds: 1.5, hopSeconds: 0.75)
        } catch {
            return Self.classified(error, module: .speakerEmbedding)
        }

        let labels = AgglomerativeClustering.cluster(
            we.segments.map { $0.embedding },
            numClusters: numSpeakers, threshold: threshold,
            maxClusters: maxSpeakers)
        let turns = Self.turnsFromWindows(we.segments, labels: labels)
        // M56 — per-request summary.
        Logger(label: AthenaLogLabel.model(.speakerEmbedding)).notice(
            """
            diarization done method=cluster \
            speakers=\(Set(labels).count) windows=\(we.segments.count) \
            turns=\(turns.count) elapsed_ms=\(Self.elapsedMs(t0))
            """)
        return Self.json(
            DiarizationResponse(
                num_speakers: Set(labels).count,
                segments: turns.map {
                    DiarizationSegmentDTO(
                        start: $0.start, end: $0.end, speaker: $0.speaker)
                }))
    }

    /// Merge time-ordered, possibly-overlapping labeled windows into
    /// contiguous same-speaker turns.
    private static func turnsFromWindows(
        _ segs: [SpeakerSegmentEmbedding], labels: [Int]
    ) -> [DiarizationTurn] {
        guard !segs.isEmpty, segs.count == labels.count else { return [] }
        var turns: [DiarizationTurn] = []
        var curLabel = labels[0]
        var curStart = segs[0].start
        var curEnd = segs[0].end
        for i in 1..<segs.count {
            if labels[i] == curLabel {
                curEnd = max(curEnd, segs[i].end)
            } else {
                turns.append(
                    DiarizationTurn(
                        start: curStart, end: curEnd, speaker: curLabel))
                curLabel = labels[i]
                curStart = segs[i].start
                curEnd = segs[i].end
            }
        }
        turns.append(
            DiarizationTurn(
                start: curStart, end: curEnd, speaker: curLabel))
        return turns
    }

    /// M25.2 — voice/speaker embeddings. Multipart `file` (audio) + an
    /// optional `segments` JSON field (`[{start,end}]`, seconds); absent
    /// ⇒ the whole clip is embedded as one segment. Returns one 256-d
    /// L2-normalized vector per segment for cross-recording speaker ID.
    private func handleSpeakerEmbeddings(_ request: Request) async
        -> Response
    {
        let t0 = Date()
        let upload = await extractUploadFile(request, cap: maxAudioUploadBytes)
        guard case .ok(let (form, file)) = upload else { return upload.orFail }

        // Optional `segments` JSON; absent ⇒ embed the whole clip.
        var segments: [SpeakerSegmentRequest] = []
        if let segText = form.text("segments"), !segText.isEmpty {
            do {
                let specs = try JSONDecoder().decode(
                    [SpeakerSegmentSpec].self,
                    from: Data(segText.utf8))
                segments = specs.map {
                    SpeakerSegmentRequest(start: $0.start, end: $0.end)
                }
            } catch {
                return Self.error(
                    status: .badRequest,
                    message: "invalid 'segments' JSON: \(error)",
                    type: "invalid_request_error",
                    code: "invalid_segments")
            }
        }

        do {
            // M43.2: non-blocking cold-load.
            switch try await governor.awaitLoad(.speakerEmbedding, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.speakerEmbedding)
            }
            // M41.3 per-request speaker-embedding model selection;
            // M41.4 audited on a real resident-id change.
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .speakerEmbedding, target: m)
            }
        } catch {
            // issue #6: classify (see textEmbedding handler) — no leak, right code.
            return Self.classified(error, module: .speakerEmbedding)
        }

        let result: SpeakerEmbeddingResult
        do {
            result = try await speakerEmbedding.embed(
                audio: file.data, filename: file.filename,
                segments: segments,
                requestedModel: form.text("model"))
        } catch {
            return Self.classified(error, module: .speakerEmbedding)
        }

        // M41.3: response.model echoes the id ACTUALLY served (post-
        // rebind), not the form-field passthrough — same truthful-echo
        // discipline as the M39 embedding batch.
        let selSpeaker = selectable(.speakerEmbedding)
        let resSpeaker = await selSpeaker.residentModelId()
        let defSpeaker = await selSpeaker.defaultModelId()
        let servedSpeaker = resSpeaker ?? defSpeaker
        // M56 — per-request summary.
        Logger(label: AthenaLogLabel.model(.speakerEmbedding)).notice(
            """
            speaker-embeddings done model=\(servedSpeaker) \
            segments=\(result.segments.count) \
            elapsed_ms=\(Self.elapsedMs(t0))
            """)
        return Self.json(
            SpeakerEmbeddingResponse(
                object: "list",
                data: result.segments.enumerated().map { idx, s in
                    SpeakerEmbeddingObject(
                        object: "speaker_embedding", index: idx,
                        segment: SpeakerSegmentSpec(
                            start: s.start, end: s.end),
                        embedding: s.embedding,
                        duration_seconds: s.durationSeconds)
                },
                model: servedSpeaker,
                dimension: result.dimension))
    }
}
