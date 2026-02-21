from langchain_core.documents import Document

chunk = Document(
    page_content=('급금이 차감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포함합니다)\n'
 '공\n'
 '계약자는 해지된 날부터 3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회\n'
 '통\n'
 '복)을 청약할 수 있습니다. 회사가 부활(효력회복)을 승낙한 때에 계약자는 부활\n'
 '(효력회복)을 청약한 날까지의 연체된 보험료에 평균공시이율 + 1% 범위내에서 각 사항\n'
 '상품별로 회사가 정하는 이율로 계산한 금액을 더하여 납입하여야 합니다. 다만 금- 리연동형보험은 각 상품별 사업방법서에서 별도로 정한 '
 '이율로 계산합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000167',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
