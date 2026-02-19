from langchain_core.documents import Document

chunk = Document(
    page_content=('. 벌과금 및 징벌적 손해에 대한 배상책임 11. 피보험자의 심신상실에 기인하는 배상책임 12. 피보험자의 지시에 따른 배상책임 13. '
 '피보험자의 불법행위 또는 폭력행위에 기인하는 배상책임'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 25},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000138',
              'chunk_char_len': 105,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
