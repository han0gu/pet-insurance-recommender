from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 계약이 갱신되는 경우에는 제1항 내지 제3항에 의한 보장은 기존 계약에 의한 보장이 종 료하는 때부터 적용합니다.\n'
 '제26조(제2회 이후 보험료의 납입)\n'
 '계약자는 제2회 이후의 보험료를 납입기일까지 납입하여야 하며, 회사는 계약자가 보험료 를 납입한 경우에는 영수증을 발행하여 드립니다. '
 '다만, 금융회사(우체국을 포함합니다)를 통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으로 대신합니다.\n'
 '【납입기일】\n'
 '계약자가 제2회 이후의 보험료를 납입하기로 한 날을 말합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 15},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000096',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
