from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 대부금의 이자는 공탁금에 붙여지는 것과 같은<br>이율로 하며, 피보험자는 공탁금(이자를 포함합니다)의 회수청구권을 회사에 '
 "양도하여<br>야 합니다.</p><h1 id='85' style='font-size:14px'>제14조(대위권)</h1><br><p "
 "id='86' data-category='list' style='font-size:14px'>① 회사가 보험금을 지급한 때(현물보상한 "
 '경우를 포함합니다)에는 회사는 지급한 보험금<br>의 한도내에서 아래의 권리를 가집니다'),
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
 'indexing': {'chunk_id': 'chunk_000254',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
