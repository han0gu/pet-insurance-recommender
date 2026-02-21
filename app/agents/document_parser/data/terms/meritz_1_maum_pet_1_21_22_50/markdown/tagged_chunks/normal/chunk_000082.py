from langchain_core.documents import Document

chunk = Document(
    page_content=('- 부분은 해지된 것으로 보며, 이로써 회사가 환급하여야 할 보험료가 있을 경우에는 제\n'
 '- 33조(보험료의 환급)에 따른 보험료를 계약자에게 지급합니다.\n'
 '- ⑤ 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경우 계약자와 피보험자가 동일하\n'
 '- 지 않을 때에는 보험금 지급사유가 발생하기 전에 피보험자가 서면(「전자서명법」 제\n'
 '- 2조 제2호에 따른 전자서명이 있는 경우로서 상법 시행령 제44조의2에 정하는 바에 따\n'
 '- 라 본인 확인 및 위조ㆍ변조 방지에 대한 신뢰성을 갖춘 전자문서를 포함)으로 동의하\n'
 '- 여야 합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000082',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
