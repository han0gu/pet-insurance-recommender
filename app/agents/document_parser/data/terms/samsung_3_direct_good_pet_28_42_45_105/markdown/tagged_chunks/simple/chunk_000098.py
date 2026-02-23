from langchain_core.documents import Document

chunk = Document(
    page_content=('- 지된 날부터 3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을 청약할 수\n'
 '- 있습니다. 회사가 부활(효력회복)을 승낙한 때에 계약자는 부활(효력회복)을 청약한\n'
 '- 날까지의 연체된 보험료에 평균공시이율+1% 범위 내에서 각 상품별로 회사가 정하는\n'
 '- 이율로 계산한 금액을 더하여 납입하여야 합니다. 다만, 금리연동형보험은 각 상품별\n'
 '- 사업방법서에서 별도로 정한 이율로 계산합니다.\n'
 '- ② 제1항에 따라 해지된 계약을 부활(효력회복)하는 경우에는 제13조(계약 전 알릴 의'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000098',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
