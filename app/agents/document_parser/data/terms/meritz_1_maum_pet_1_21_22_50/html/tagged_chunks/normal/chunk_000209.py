from langchain_core.documents import Document

chunk = Document(
    page_content=('손해의 배상을 받을 수 있는 경우에 피보험자가 손해배상청구를 위<br>해 내용증명, 재산조사, 강제집행 등을 수행하고자 지출한 각종 '
 "비용을 의미합니<br>다.</p><br><p id='28' data-category='list' "
 "style='font-size:14px'>다"),
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
 'indexing': {'chunk_id': 'chunk_000209',
              'chunk_char_len': 152,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
