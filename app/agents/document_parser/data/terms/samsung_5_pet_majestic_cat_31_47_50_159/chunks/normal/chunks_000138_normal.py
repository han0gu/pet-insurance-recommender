from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 부활(효력회복)을 승낙한 때에 계약자는 부활(효력회복)을 청약한 날까지의 연체된 보험료에 평균공시이율+1% 범위 내에서 각 '
 '상품별로 회사가 정하는 이율로 계산한 금액을 더하여 납입하여야 합니다. 다만, 금리연동형보험은 각 상품별 사업방법서에서 별도로 정한 '
 '이율로 계산합니다. ② 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는 제16조(계약 전 알릴 의무), 제18조(알릴 의무 위반의 '
 '효과), 제19조(사기에 의한 계약), 제20조(보험계약의 성립) 및 제27조(제1회 보험료 및 회사의 보장개시)를 준용합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 43},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000138',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
