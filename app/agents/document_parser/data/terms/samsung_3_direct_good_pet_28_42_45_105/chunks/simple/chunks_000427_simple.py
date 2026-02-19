from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 부활(효력회복)을 승낙한 때에는 계약자는 부활(효 력회복)을 청약한 날까지의 연체된 보험료에 평균공시이율 + 1% 범위 내에서 '
 '각 상품 별로 회사가 정하는 이율로 계산한 금액을 더하여 납입하여야 합니다. 다만, 금리연동 형보험은 각 상품별 사업방법서에서 별도로 '
 '정한 이율로 계산합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000427',
              'chunk_char_len': 164,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
