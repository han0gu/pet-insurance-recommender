from langchain_core.documents import Document

chunk = Document(
    page_content=('급하여야 할 보험료가 있을 경우에는 제33조(보험료의 환급)에 따른 보험료를 계약자에게\n'
 '지급합니다.제30조의2(위법계약의 해지)① 계약자는 ｢금융소비자보호에 관한 법률｣ 제47조 및 관련규정이 정하는 바에 따라 계약\n'
 '체결에 대한 회사의 법위반사항이 있는 경우 계약체결일부터 5년 이내의 범위에서 계\n'
 '약자가 위반사항을 안 날부터 1년 이내에 계약해지요구서에 증빙서류를 첨부하여 위법\n'
 '계약의 해지를 요구할 수 있습니다.\n'
 '② 회사는 해지요구를 받은 날부터 10일 이내에 수락여부를 계약자에 통지하여야 하며, 거'),
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
 'indexing': {'chunk_id': 'chunk_000097',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
