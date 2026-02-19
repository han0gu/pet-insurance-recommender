from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 한쪽 폐 또는 한쪽 신장을 전부 잘라내었을 때 나) 방광 기능상실로 영구적인 요도루, 방광루, 요관 장문합 상태 다) 위, 췌장을 '
 '50% 이상 잘라내었을 때 라) 대장절제, 항문 괄약근 등의 기능장해로 영구적으'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 199},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['other', 'urinary', 'digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000726',
              'chunk_char_len': 120,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
