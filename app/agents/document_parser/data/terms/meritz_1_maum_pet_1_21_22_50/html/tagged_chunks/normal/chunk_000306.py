from langchain_core.documents import Document

chunk = Document(
    page_content=("사항은 보통약관 및 보험료자동납입 특별약관을 따릅니다.</p><footer id='47' style='font-size:14px'>- "
 "36 -</footer><h1 id='48' style='font-size:18px'>단체계약 특별약관</h1><h1 id='49' "
 "style='font-size:14px'>제1조(계약의 적용 범위)</h1><br><p id='50' "
 "data-category='paragraph' style='font-size:14px'>① 피보험자가 다음 중 한가지의 단체에 소속되어야 "
 '하며, 단체를 대표하여'),
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
 'indexing': {'chunk_id': 'chunk_000306',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
