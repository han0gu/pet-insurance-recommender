from langchain_core.documents import Document

chunk = Document(
    page_content=('- 수 있습니다.\n'
 '- \uf000 제1항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게\n'
 '- 돌려드립니다.\n'
 '- 제11조(보험계약의 성립)\n'
 '- \uf000 계약은 계약자의 청약과 회사의 승낙으로 이루어집니다.\n'
 '- 특\n'
 '- \uf000 회사는 반려동물이 이 특별약관에 적합하지 않은 경우에는 승낙을 거절하거나\n'
 '- 별\n'
 '- 별도의 조건(보험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증\n'
 '- 약\n'
 '- 등)을 붙여 승낙할 수 있습니다.\n'
 '- 관\n'
 '- \uf000 회사는 계약의 청약을 받고, 제1회 보험료를 받은 경우에 건강진단을 받지 않'),
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
 'indexing': {'chunk_id': 'chunk_000500',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
