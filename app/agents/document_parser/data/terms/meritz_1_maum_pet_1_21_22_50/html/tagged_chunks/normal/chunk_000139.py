from langchain_core.documents import Document

chunk = Document(
    page_content=("이 약관이 정하는 바에 따라 보장을 합니다.</p><br><h1 id='43' "
 "style='font-size:14px'>【보장개시일】</h1><br><p id='44' data-category='paragraph' "
 "style='font-size:14px'>회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 보험료를 받은 날을 말하나,<br>회사가 "
 '승낙하기 전이라도 청약과 함께 제1회 보험료를 받은 경우에는 제1회 보험료<br>를 받은 날을 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000139',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
