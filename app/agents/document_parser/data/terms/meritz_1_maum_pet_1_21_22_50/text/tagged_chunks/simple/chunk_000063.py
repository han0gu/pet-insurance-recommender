from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 회사는 계약의 청약을 받고, 제1회 보험료를 받은 경우에 건강진단을 받지 않는 계약은\n'
 '청약일, 진단계약은 진단일(재진단의 경우에는 최종 진단일)부터 30일 이내에 승낙 또\n'
 '는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니다. 그러나 30일 이내에 승낙 또\n'
 '는 거절의 통지가 없으면 승낙된 것으로 봅니다.\n'
 '④ 회사가 제1회 보험료를 받고 승낙을 거절한 경우에는 거절통지와 함께 받은 금액을 계\n'
 '약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 ‘보험개발원이 공시하는 월평균'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000063',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
