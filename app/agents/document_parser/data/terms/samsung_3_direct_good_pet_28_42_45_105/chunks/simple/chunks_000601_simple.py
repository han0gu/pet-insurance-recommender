from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑤ 제1항의 경우 피보험자가 병원 또는 의원을 이전하여 입원한 경우에도 동일한 질병의 치료를 직접 목적으로 입원한 경우에는 계속하여 '
 '입원한 것으로 보아 각 입원일수를 더합니다. ⑥ 제4항에도 불구하고 동일한 질병에 대한 입원이라도 반려견 위탁비용이 지급된 최종 입원의 '
 '퇴원일로부터 180일이 경과하여 개시한 입원은 새로운 입원으로 간주합니다. 다만, 아래와 같이 반려견 위탁비용이 지급된 최종입원일부터 '
 '180일이 경과하도록 퇴 원없이 계속 입원중인 경우에는 반려견 위탁비용이 지급된 최종 입원일의 그 다음날 을 퇴원일로 봅니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000601',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
