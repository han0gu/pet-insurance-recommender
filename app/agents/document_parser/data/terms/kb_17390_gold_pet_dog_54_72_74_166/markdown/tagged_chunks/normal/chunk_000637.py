from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 이 특별약관의 【갱신계약】은 "제도성 특별약관 - 보장특약 자동갱신(추가납입\n'
 '- 형) 특별약관"에 의해 계약자의 선택에 따라 자동갱신으로 운영합니다.\n'
 '\uf000 제1항에 의해 자동갱신을 적용할 경우 보험증권에 그 내용을 기재하여 드립니다.제7조(보험료의 납입을 연체하여 해지된 계약의 '
 '부활(효력회복))\n'
 '부활(효력회복)되는 계약의 보장개시는 반려동물(강아지) 일반조항 제17조(보험료'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000637',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
