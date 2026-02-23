from langchain_core.documents import Document

chunk = Document(
    page_content=('료하는 때부터 적용합니다.# 제26조(제2회 이후 보험료의 납입)계약자는 제2회 이후의 보험료를 납입기일까지 납입하여야 하며, 회사는 '
 '계약자가 보험료\n'
 '를 납입한 경우에는 영수증을 발행하여 드립니다. 다만, 금융회사(우체국을 포함합니다)를'),
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
 'indexing': {'chunk_id': 'chunk_000087',
              'chunk_char_len': 132,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
