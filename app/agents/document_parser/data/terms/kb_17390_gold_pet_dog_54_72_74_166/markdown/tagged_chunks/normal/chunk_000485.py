from langchain_core.documents import Document

chunk = Document(
    page_content=('↓\n'
 '계약변경사항 인수 심사\n'
 '↓\n'
 '정산금액 처리(환급 또는 추가납입)\n'
 '↓\n'
 '계약변경 완료- \uf000 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소된 경우에는 보험료를 감\n'
 '- 액하고, 이후 기간 보장을 위한 재원인 계약자적립액 등의 차이로 인하여 발생한\n'
 '- 정산금액(이하 "정산금액"이라 합니다)을 환급하여 드립니다. 한편 위험이 증가\n'
 '- 된 경우에는 보험료의 증액 및 정산금액의 추가납입을 요구할 수 있으며, 계약자\n'
 '- 는 이를 납입하여야 합니다.\n'
 '- \uf000 제1항의 통지에 따라 위험의 증가로 보험료를 더 내야 할 경우 회사가 청구한 추'),
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
 'indexing': {'chunk_id': 'chunk_000485',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
