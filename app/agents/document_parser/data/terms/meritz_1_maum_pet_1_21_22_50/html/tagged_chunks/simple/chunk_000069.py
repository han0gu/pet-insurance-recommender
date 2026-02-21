from langchain_core.documents import Document

chunk = Document(
    page_content=("부담한 총 비용금액</h1><br><p id='77' data-category='paragraph' "
 "style='font-size:14px'>×</p><br><p id='78' data-category='paragraph' "
 "style='font-size:14px'>이 계약의 지급보험금</p><br><p id='79' "
 "data-category='paragraph' style='font-size:14px'>다른 계약이 없는 것으로 하여 "
 "각각</p><br><p id='80' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000069',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
