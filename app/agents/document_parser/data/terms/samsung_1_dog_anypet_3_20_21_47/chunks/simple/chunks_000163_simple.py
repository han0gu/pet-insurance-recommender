from langchain_core.documents import Document

chunk = Document(
    page_content=('치료비보상 제외 특별약관\n'
 '제1 조(보험금을 지급하지 않는 사유)\n'
 '회사는 보통약관 제4조(보상하는 손해)에도 불구하고 상해 또는 질병 치료비로 인한 보험금은 이 특별 약관에 따라 보상하지 않습니다.\n'
 '제2조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 31},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000163',
              'chunk_char_len': 150,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
