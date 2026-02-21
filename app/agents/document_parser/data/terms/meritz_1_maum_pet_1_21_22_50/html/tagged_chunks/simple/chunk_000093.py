from langchain_core.documents import Document

chunk = Document(
    page_content=("의무 위반의 효과)</h1><br><p id='113' data-category='paragraph' "
 "style='font-size:14px'>① 회사는 아래와 같은 사실이 있을 경우에는 보험금 지급사유의 발생여부에 관계없이 "
 "그<br>사실을 안 날부터 1개월 이내에 이 계약을 해지할 수 있습니다.</p><br><p id='114' "
 "data-category='list' style='font-size:14px'>1"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000093',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
