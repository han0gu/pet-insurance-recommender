from langchain_core.documents import Document

chunk = Document(
    page_content=('약자는 해지된 날부터 3년 이내에 회사가 정한 절차에 따라 특별약관의 부활(효력회 복)을 청약할 수 있습니다. 회사가 부활(효력회복)을 '
 '승낙한 때에는 계약자는 부활(효 력회복)을 청약한 날까지의 연체된 보험료에 평균공시이율 + 1% 범위 내에서 각 상품 별로 회사가 정하는 '
 '이율로 계산한 금액을 더하여 납입하여야 합니다. 다만, 금리연동 형보험은 각 상품별 사업방법서에서 별도로 정한 이율로 계산합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000619',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
