import logging
from importlib import metadata
from datetime import datetime
import cupy as cp
from exorde.models import (
    Domain,
    ProtocolItem,
    ProtocolAnalysis,
    ProcessedItem,
    Batch,
    BatchKindEnum,
    CollectionClientVersion,
    CollectedAt,
    CollectionModule,
    Processed,
    Analysis,
)
from exorde.models import (
    Classification,
    Keywords,
    LanguageScore,
    Sentiment,
    Embedding,
    SourceType,
    TextType,
    Emotion,
    Irony,
    Age,
    Gender,
    Analysis
)
from exorde_data import Url, Content

from exorde.tag import tag
from collections import Counter


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


import cupy as cp

def merge_chunks(chunks: list[ProcessedItem]) -> ProcessedItem:
    try:
        if len(chunks) == 1:
            return chunks[0]

        categories_list = []
        top_keywords_list = []
        gender_list = []
        sentiment_list = []
        source_type_list = []
        text_type_list = []
        emotion_list = []
        language_score_list = []
        irony_list = []
        age_list = []
        embedding_list = []

        logging.info(f"[Item merging] Merging {len(chunks)} chunks.")
        for processed_item in chunks:
            item_analysis_ = processed_item.analysis
            categories_list.append(item_analysis_.classification)
            top_keywords_list.append(item_analysis_.top_keywords)
            gender_list.append(item_analysis_.gender)
            sentiment_list.append(item_analysis_.sentiment)
            source_type_list.append(item_analysis_.source_type)
            text_type_list.append(item_analysis_.text_type)
            emotion_list.append(item_analysis_.emotion)
            language_score_list.append(item_analysis_.language_score)
            irony_list.append(item_analysis_.irony)
            age_list.append(item_analysis_.age)
            embedding_list.append(item_analysis_.embedding)

        # Aggregating the values
        most_common_category = Most_Common([x.label for x in categories_list])
        category_aggregated = Classification(
            label=most_common_category,
            score=max([x.score for x in categories_list]),
        )

        top_keywords_aggregated = list()
        for top_keywords in top_keywords_list:
            top_keywords_aggregated.extend(top_keywords)
        top_keywords_aggregated = Keywords(
            list(set(top_keywords_aggregated))
        )

        gender_aggregated = Gender(
            male=cp.median(cp.array([x.male for x in gender_list])),
            female=cp.median(cp.array([x.female for x in gender_list])),
        )

        sentiment_aggregated = Sentiment(cp.median(cp.array(sentiment_list)))

        source_type_aggregated = SourceType(
            Most_Common(source_type_list)
        )

        text_type_aggregated = TextType(
            assumption=cp.median(cp.array([tt.assumption for tt in text_type_list])),
            anecdote=cp.median(cp.array([tt.anecdote for tt in text_type_list])),
            none=cp.median(cp.array([tt.none for tt in text_type_list])),
            definition=cp.median(cp.array([tt.definition for tt in text_type_list])),
            testimony=cp.median(cp.array([tt.testimony for tt in text_type_list])),
            other=cp.median(cp.array([tt.other for tt in text_type_list])),
            study=cp.median(cp.array([tt.study for tt in text_type_list])),
        )

        emotion_aggregated = Emotion(
            love=cp.median(cp.array([e.love for e in emotion_list])),
            admiration=cp.median(cp.array([e.admiration for e in emotion_list])),
            joy=cp.median(cp.array([e.joy for e in emotion_list])),
            approval=cp.median(cp.array([e.approval for e in emotion_list])),
            caring=cp.median(cp.array([e.caring for e in emotion_list])),
            excitement=cp.median(cp.array([e.excitement for e in emotion_list])),
            gratitude=cp.median(cp.array([e.gratitude for e in emotion_list])),
            desire=cp.median(cp.array([e.desire for e in emotion_list])),
            anger=cp.median(cp.array([e.anger for e in emotion_list])),
            optimism=cp.median(cp.array([e.optimism for e in emotion_list])),
            disapproval=cp.median(cp.array([e.disapproval for e in emotion_list])),
            grief=cp.median(cp.array([e.grief for e in emotion_list])),
            annoyance=cp.median(cp.array([e.annoyance for e in emotion_list])),
            pride=cp.median(cp.array([e.pride for e in emotion_list])),
            curiosity=cp.median(cp.array([e.curiosity for e in emotion_list])),
            neutral=cp.median(cp.array([e.neutral for e in emotion_list])),
            disgust=cp.median(cp.array([e.disgust for e in emotion_list])),
            disappointment=cp.median(cp.array([e.disappointment for e in emotion_list])),
            realization=cp.median(cp.array([e.realization for e in emotion_list])),
            fear=cp.median(cp.array([e.fear for e in emotion_list])),
            relief=cp.median(cp.array([e.relief for e in emotion_list])),
            confusion=cp.median(cp.array([e.confusion for e in emotion_list])),
            remorse=cp.median(cp.array([e.remorse for e in emotion_list])),
            embarrassment=cp.median(cp.array([e.embarrassment for e in emotion_list])),
            surprise=cp.median(cp.array([e.surprise for e in emotion_list])),
            sadness=cp.median(cp.array([e.sadness for e in emotion_list])),
            nervousness=cp.median(cp.array([e.nervousness for e in emotion_list])),
        )

        language_score_aggregated = LanguageScore(
            cp.median(cp.array(language_score_list))
        )

        irony_aggregated = Irony(
            irony=cp.median(cp.array([i.irony for i in irony_list])),
            non_irony=cp.median(cp.array([i.non_irony for i in irony_list])),
        )

        age_aggregated = Age(
            below_twenty=cp.median(cp.array([a.below_twenty for a in age_list])),
            twenty_thirty=cp.median(cp.array([a.twenty_thirty for a in age_list])),
            thirty_forty=cp.median(cp.array([a.thirty_forty for a in age_list])),
            forty_more=cp.median(cp.array([a.forty_more for a in age_list])),
        )

        centroid_vector = cp.median(cp.array(embedding_list), axis=0)

        closest_embedding = Embedding(
            min(
                embedding_list,
                key=lambda x: cp.linalg.norm(cp.array(x) - centroid_vector),
            )
        )

        merged_item = ProcessedItem(
            item=chunks[0].item,
            analysis=ProtocolAnalysis(
                classification=category_aggregated,
                top_keywords=top_keywords_aggregated,
                language_score=language_score_aggregated,
                gender=gender_aggregated,
                sentiment=sentiment_aggregated,
                embedding=closest_embedding,
                source_type=source_type_aggregated,
                text_type=text_type_aggregated,
                emotion=emotion_aggregated,
                irony=irony_aggregated,
                age=age_aggregated,
            ),
            collection_client_version=chunks[0].collection_client_version,
            collection_module=chunks[0].collection_module,
            collected_at=chunks[0].collected_at,
        )
    except Exception as e:
        logging.exception(f"[Merging items chunks] ERROR:\n {e}")
        merged_item = None
    return merged_item

SOCIAL_DOMAINS = [
    "4chan",
    "4channel.org",
    "reddit.com",
    "twitter.com",
    "bsky.app",
    "t.com",
    "x.com",
    "youtube.com",
    "yt.co",
    "lemmy.world",
    "mastodon.social",
    "mastodon",
    "weibo.com",
    "nostr.social",
    "nostr.com",
    "jeuxvideo.com",
    "forocoches.com",
    "bitcointalk.org",
    "ycombinator.com",
    "news.ycombinator.com",
    "tradingview.com",
    "followin.in",
    "seekingalpha.io"
]

def get_source_type(item: ProtocolItem) -> SourceType:
    if item.domain in SOCIAL_DOMAINS:
        return SourceType("social")
    return SourceType("news")


async def process_batch(
    batch: list[tuple[int, Processed]], static_configuration
) -> Batch:
    lab_configuration: dict = static_configuration["lab_configuration"]
    logging.info(f"running batch for {len(batch)}")
    analysis_results: list[Analysis] = tag(
        [processed.translation.translation for (__id__, processed) in batch],
        lab_configuration,
    )
    complete_processes: dict[int, list[ProcessedItem]] = {}
    for (id, processed), analysis in zip(batch, analysis_results):
        prot_item: ProtocolItem = ProtocolItem(
            raw_content=Content(processed.raw_content),
            translated_content=Content(processed.translation.translation),
            created_at=processed.item.created_at,
            domain=processed.item.domain,
            url=Url(processed.item.url),
            language=processed.translation.language,
        )

        if processed.item.title:
            prot_item.title = processed.item.title
        if processed.item.summary:
            prot_item.summary = processed.item.summary
        if processed.item.picture:
            prot_item.picture = processed.item.picture
        if processed.item.author:
            prot_item.author = processed.item.author
        if processed.item.external_id:
            prot_item.external_id = processed.item.external_id
        if processed.item.external_parent_id:
            prot_item.external_parent_id = processed.item.external_parent_id
        completed: ProcessedItem = ProcessedItem(
            item=prot_item,
            analysis=ProtocolAnalysis(
                classification=processed.classification,
                top_keywords=processed.top_keywords,
                language_score=analysis.language_score,
                gender=analysis.gender,
                sentiment=analysis.sentiment,
                embedding=analysis.embedding,
                source_type=get_source_type(prot_item),
                text_type=analysis.text_type,
                emotion=analysis.emotion,
                irony=analysis.irony,
                age=analysis.age,
            ),
            collection_client_version=CollectionClientVersion(
                f"exorde:v.{metadata.version('exorde_data')}"
            ),
            collection_module=CollectionModule("unknown"),
            collected_at=CollectedAt(datetime.now().isoformat() + "Z"),
        )
        if not complete_processes.get(id, {}):
            complete_processes[id] = []
        complete_processes[id].append(completed)
    aggregated = []
    for __key__, values in complete_processes.items():
        merged_ = merge_chunks(values)
        if merged_ is not None:
            aggregated.append(merged_)
    result_batch: Batch = Batch(items=aggregated, kind=BatchKindEnum.SPOTTING)
    return result_batch
