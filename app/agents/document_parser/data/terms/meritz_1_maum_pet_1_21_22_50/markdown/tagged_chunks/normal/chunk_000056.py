from langchain_core.documents import Document

chunk = Document(
    page_content=('- 료의 증액 및 정산금액의 추가납입을 요구할 수 있으며, 계약자는 일시납 또는 잔여 보험\n'
 '- 료 납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초과할 수 없음) 동안의 분납 중 선택\n'
 '- 하여 정산금액을 납입하여야 합니다. 다만, 보험료 갱신형 계약 등 일부 보험계약의 경우\n'
 '- 분납이 제한될 수 있습니다.\n'
 '- ④ 제1항의 통지에 따라 위험의 증가로 보험료를 더 내야 할 경우 회사가 청구한 추가보\n'
 '- 험료(정산금액을 포함합니다)를 계약자가 납입하지 않았을 때, 회사는 위험이 증가되기'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000056',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
