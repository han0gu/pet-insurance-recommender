from langchain_core.documents import Document

chunk = Document(
    page_content=('이상의 피보험자로<br>피보험단체를 구성하여야 하며, 단체 구성원의 일부만을 대상으로 가입하는 경우에는<br>다음의 조건을 모두 '
 "충족하여야 합니다.</p><br><p id='58' data-category='list' style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_000311',
              'chunk_char_len': 146,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
