from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 보험의 목적이 다수인 경우 제1항은 보험의 목적별로 각각 적용합니다.\n'
 '- 반\n'
 '및제6조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))부활(효력회복)되는 계약의 보장개시는 반려동물(강아지) 일반조항려동의 '
 '납입을 연체하여 해지된 특별약관의 부활(효력회복))를 따릅니다. 이 경우 부활\n'
 '(효력회복)일을 보험계약일로 하여 제1조(보험금의 지급사유) 제4항 내지 제5항을\n'
 '적용합니다.\n'
 '제'),
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
 'indexing': {'chunk_id': 'chunk_000570',
              'chunk_char_len': 220,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
