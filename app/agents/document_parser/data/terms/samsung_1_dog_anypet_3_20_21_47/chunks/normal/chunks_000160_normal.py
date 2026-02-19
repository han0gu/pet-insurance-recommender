from langchain_core.documents import Document

chunk = Document(
    page_content=('. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태 10. 원인이 어떠한 경우에도 반려동물에 대한 사료제공 '
 '또는 급수 등 기본적인 관리에 대한 태만'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 30},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000160',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
