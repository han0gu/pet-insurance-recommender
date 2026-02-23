from langchain_core.documents import Document

chunk = Document(
    page_content=("다음 1년의 원금으로 하는 이자 계산방법을 말합니다.<br>원금 100원, 이자율 연 10%를 가정할 때</p><br><p id='88' "
 "data-category='list' style='font-size:14px'>- 1년 후 원리금 : 100원 + (100원×10%) = "
 "110원<br>- 2년 후 원리금 : 110원 + (110원×10%) = 121원</p><h1 id='89' "
 "style='font-size:14px'>제12조(주소변경통지)</h1><br><p id='90' data-category='list'"),
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
 'indexing': {'chunk_id': 'chunk_000074',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
