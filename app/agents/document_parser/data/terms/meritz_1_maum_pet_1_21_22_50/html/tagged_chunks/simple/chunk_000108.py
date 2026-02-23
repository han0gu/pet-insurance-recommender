from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 30일 이내에 승낙 또<br>는 거절의 통지가 없으면 승낙된 것으로 봅니다.<br>④ 회사가 제1회 보험료를 받고 승낙을 '
 '거절한 경우에는 거절통지와 함께 받은 금액을 계<br>약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 ‘보험개발원이 공시하는 '
 '월평균<br>정기예금이율 + 1%’를 연단위 복리로 계산한 금액을 더하여 지급합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000108',
              'chunk_char_len': 192,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
