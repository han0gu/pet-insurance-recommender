from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만 금리연동형보험은 각 상품별 사업방법서에서 별도로 정한 이율로 계<br>산합니다.<br>\uf000 제1항에 따라 해지된 '
 '특별약관을 부활(효력회복)하는 경우에는 제7조(계약 전 알<br>릴의무), 제9조(알릴 의무 위반의 효과), 제10조(사기에 의한 '
 '계약), 제15조(제1<br>회 보험료 및 회사의 보장개시), 보통약관 제1절 일반조항 제18조(보험계약의 성<br>립)를 준용합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000892',
              'chunk_char_len': 214,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
