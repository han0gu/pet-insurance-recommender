from langchain_core.documents import Document

chunk = Document(
    page_content=('제집행 등으로 인하여 해지된 특별약관의 특별부활(효력회복))에 따라 이 특별약관의\n'
 '부활(효력회복)을 취급합니다.청약을 받은 경우에는 보험계약의 부활(효력회복)- 139 -제4조(준용규정)이 특별약관에 정하지 아니한 '
 '사항에 대하여는 보통약관 및 해당 특별약관의 규정을따릅니다.별특약관상해질병상해및질병반려동물제도성특약KB 금쪽같은 '
 '펫보험(강아지)(무배당)(26.01) 139|  | 별표 |\n'
 '| --- | --- |\n'
 '# 별표1 장해분류표# \uf000 총칙- 1. 장해의 정의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000829',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
