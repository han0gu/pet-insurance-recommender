from langchain_core.documents import Document

chunk = Document(
    page_content=('려동- \n'
 '질제8조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))병부활(효력회복)되는 계약의 보장개시는 반려동물(강아지) 일반조항의 '
 '납입을 연체하여 해지된 특별약관의 부활(효력회복))를 따릅니다. 이 경우 부활\n'
 '(효력회복)일을 보험계약일로 하여 제1조(보험금의 지급사유) 제6항을 적용합니다.\n'
 '제\n'
 '도# 제9조(준용규정)제17조(보험료물# \uf000 이 특별약관에서정하지 않은 사항은 반려동물(강아지) 일반조항을 따릅니다. 다만, 이 '
 '특별약관에서는 반려동물(강아지) 일반조항 제22조(재가입)은 제외합니'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000655',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
