from langchain_core.documents import Document

chunk = Document(
    page_content=('제1항에 따라 해지계약을 부활(효력회복)하는 경우에는 제12조(계약 전 알릴의무), 제14조(사기에 의한 계약), 제15조(보험계약의 '
 '성립), 제21조(제1회 보험료 등 및 회사의 보장개시) 및 제26조(계 약의 해지)의 규정을 준용합니다. 이 때 회사는 해지 전 발생한 '
 '보험금 지급사유를 이유로 부활(효 력회복)을 거절하지 않습니다'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 15},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000074',
              'chunk_char_len': 186,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
