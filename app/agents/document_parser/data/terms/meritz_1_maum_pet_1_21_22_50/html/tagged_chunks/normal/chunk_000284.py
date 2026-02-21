from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>- 30 -</footer><h1 id='0' style='font-size:18px'>반려견 "
 "슬관절·고관절 치료비 보장 특별약관</h1><p id='1' data-category='paragraph' "
 "style='font-size:14px'>제1조(보상하는 손해)</p><br><p id='2' data-category='list' "
 "style='font-size:14px'>① 회사는 보통약관 제5조(보험금을 지급하지 않은 사유) 제2항 제3호에도 "
 '불구하고<br>슬관절탈구, 고관절탈구,'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000284',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
