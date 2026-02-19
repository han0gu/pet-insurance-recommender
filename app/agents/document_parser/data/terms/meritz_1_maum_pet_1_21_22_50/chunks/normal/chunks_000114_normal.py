from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자는 ｢금융소비자보호에 관한 법률｣ 제47조 및 관련규정이 정하는 바에 따라 계약 체결에 대한 회사의 법위반사항이 있는 경우 '
 '계약체결일부터 5년 이내의 범위에서 계 약자가 위반사항을 안 날부터 1년 이내에 계약해지요구서에 증빙서류를 첨부하여 위법 계약의 해지를 '
 '요구할 수 있습니다. ② 회사는 해지요구를 받은 날부터 10일 이내에 수락여부를 계약자에 통지하여야 하며, 거 절할 때에는 거절 사유를 '
 '함께 통지하여야 합니다. ③ 계약자는 회사가 정당한 사유 없이 제1항의 요구를 따르지 않는 경우 해당 계약을 해 지할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 18},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000114',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
