from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 회사는 그러한 보증을 제공할 책임은 부담하지 않습니다. 마. 피보험자가 제8조(손해배상청구에 대한 회사의 해결) 제2항 및 '
 '제3항의 회사의 요구에 따르 기 위하여 지출한 비용'),
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
 'indexing': {'chunk_id': 'chunk_000134',
              'chunk_char_len': 103,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
