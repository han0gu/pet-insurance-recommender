from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조(갱신보장특약 제1회 보험료의 납입최고(독촉)와 계약의 해지)\uf000 계약자는 보통약관 제1절 일반조항 제28조(보험료의 납입이 '
 '연체되는 경우반납입최- 려\n'
 '- 고(독촉)와 계약의 해지)에 정한 납입최고(독촉)기간 내에 갱신 전 보장계약의 보\n'
 '- 동\n'
 '- 험료를 납입 완료하고, 제2조(보장특약의 자동갱신)에 의해 보장특약이 자동 갱신 물\n'
 '- 된 경우에는 갱신보장특약의 제1회 보험료를 갱신 일까지 납입하여야 합니다.\n'
 '- \uf000 제1항에도 불구하고 계약자가 갱신 일까지 갱신보장특약의 제1회 보험료를 납입하'),
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
 'indexing': {'chunk_id': 'chunk_000787',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
