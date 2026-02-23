from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 계약의 청약을 받고, 제1회 보험료를 받은 경우에 건강진단을 받지 않\n'
 '- 는 계약은 청약일, 진단계약은 진단일(재진단의 경우에는 최종 진단일)부터 30\n'
 '- 일 이내에 승낙 또는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니다.\n'
 '- 그러나 30일 이내에 승낙 또는 거절의 통지가 없으면 승낙된 것으로 봅니다.\n'
 '- \uf000 회사가 제1회 보험료를 받고 승낙을 거절한 경우에는 거절통지와 함께 받은 금 상\n'
 '- 액을 계약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 평균공시이율 + 해'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000501',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
