from langchain_core.documents import Document

chunk = Document(
    page_content=('약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 ‘보험개발원이 공시하는 월평균\n'
 '정기예금이율 + 1%’를 연단위 복리로 계산한 금액을 더하여 지급합니다. 다만, 회사는\n'
 '계약자가 제1회 보험료를 신용카드로 납입한 계약의 승낙을 거절하는 경우에는 신용카\n'
 '드의 매출을 취소하며 이자를 더하여 지급하지 않습니다.제20조(청약의 철회)① 계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 '
 '철회할 수 있습니다. 다만,\n'
 '회사가 건강상태 진단을 지원하는 계약, 보험기간이 90일 이내인 계약 또는 전문금융소'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000064',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
