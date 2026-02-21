from langchain_core.documents import Document

chunk = Document(
    page_content=('적용합니다.\n'
 '제\n'
 '도제17조(보험료# 제7조(준용규정)# \uf000 이 특별약관에서정하지 않은 사항은 반려동물(강아지) 일반조항을 따릅니다.물\uf000 '
 '반려동물(강아지) 일반조항에서 정하지 않은 사항은 보통약관 제1절 일반조항을성특KB 금쪽같은 펫보험(강아지)(무배당)(26.01) '
 '111- 111 -약따릅니다. 다만, 이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계약의 소멸) 및 '
 '제36조(중도인출)는 제외합니다.# 2. 반려동물의료비확장보장Ⅱ(주요치료)(강아지)제1조(보험금의 지급사유)'),
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
 'indexing': {'chunk_id': 'chunk_000571',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
