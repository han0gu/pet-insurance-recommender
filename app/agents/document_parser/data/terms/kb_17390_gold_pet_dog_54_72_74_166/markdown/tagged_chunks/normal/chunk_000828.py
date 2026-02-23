from langchain_core.documents import Document

chunk = Document(
    page_content=('- 정합니다.\n'
 '- \uf000 제1항의 특정질병은 1개에 한하여 부가할 수 있습니다.\n'
 '제3조(해지된 특별약관의 부활(효력회복))# 회사는 이 특약의 부활(효력회복)을 승낙한 경우에 한하여 제4장 반려동물 관련 특별약관 '
 '반려동물(강아지) 일반조항\n'
 '제17조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복)) 및 제18조(강\n'
 '제집행 등으로 인하여 해지된 특별약관의 특별부활(효력회복))에 따라 이 특별약관의'),
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
 'indexing': {'chunk_id': 'chunk_000828',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
