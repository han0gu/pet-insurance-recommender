from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 지정대리청구인은 보험금 청구 시에도 다음 각 호의 1에 해당하여야 합니<br>다.</p><br><p id='17' "
 "data-category='list' style='font-size:14px'>1. 피보험자의 가족관계등록부상 또는 주민등록상의 "
 '배우자<br>2'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000347',
              'chunk_char_len': 150,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
