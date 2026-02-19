from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항의 납입최고(독촉)기간은 납입최고(독촉)의 통지가 계약자(타인을 위한 계약의 경우 그 특정 된 타인을 포함합니다)에게 도달한 '
 '날부터 시작되며, 납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그 다음 날까지로 합니다. ③ 회사가 제1항에 '
 '의한 납입최고(독촉) 등을 전자문서로 안내하고자 할 경우에는 계약자의 서면에 의 한 동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 '
 '하며, 계약자가 전자문서에 대하여 수신을 확인하기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 14},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000072',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
