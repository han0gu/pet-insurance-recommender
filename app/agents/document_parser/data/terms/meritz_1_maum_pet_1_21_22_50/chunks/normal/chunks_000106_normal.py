from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는 제15조(계약 전 알릴 의무), 제 17조(알릴 의무 위반의 효과), '
 '제18조(사기에 의한 계약), 제19조(보험계약의 성립) 및 제25조(제1회 보험료 및 회사의 보장개시)의 규정을 준용합니다. 이 때 '
 '회사는 해지 전 발생한 보험금 지급사유를 이유로 부활(효력회복)을 거절하지 않습니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000106',
              'chunk_char_len': 192,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
