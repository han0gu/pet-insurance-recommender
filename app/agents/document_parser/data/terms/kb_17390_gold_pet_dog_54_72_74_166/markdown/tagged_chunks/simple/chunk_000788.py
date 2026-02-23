from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항에도 불구하고 계약자가 갱신 일까지 갱신보장특약의 제1회 보험료를 납입하\n'
 '- 지 않는 때에는 보통약관 제1절 일반조항 제28조(보험료의 납입이 연체되는 경우\n'
 '- 제\n'
 '- 납입최고(독촉)와 계약의 해지)에 따라 납입최고(독촉)하며, 이 납입최고(독촉)\n'
 '- 도\n'
 '- 기간 안에 보험료를 납입하지 않는 경우 납입최고(독촉)기간이 끝나는 날의 다음\n'
 '- 날 해당 보장특약은 해제된 것으로 봅니다. 성특\n'
 '- \uf000 회사는 납입최고(독촉)기간 안에 발생한 사고에 대하여 약정한 보험금을 지급합니 약'),
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
 'indexing': {'chunk_id': 'chunk_000788',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
