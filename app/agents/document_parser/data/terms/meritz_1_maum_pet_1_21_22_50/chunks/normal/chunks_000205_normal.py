from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 단체의 내규에 의한 복지제도로서 노사합의에 의하며, 보험료의 일부를 단체 또는 단 체의 대표자가 부담하여야 합니다. 2. 제1항 '
 '제2호 및 제3호에 해당하는 단체는 내규에 의해 단체의 대표자와 보험회사가 협정에 의해 체결하여야 합니다.\n'
 '제2조(상법 제735조3의 적용)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 37},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000205',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
