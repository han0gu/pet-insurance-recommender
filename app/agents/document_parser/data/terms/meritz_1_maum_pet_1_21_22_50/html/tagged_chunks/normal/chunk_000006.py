from langchain_core.documents import Document

chunk = Document(
    page_content=('투견, 경주견 등 흥행을 목적으로 사육ㆍ관리하는 개(犬) 또는 흥행을 목적으로<br>사육ㆍ관리하는 고양이(猫)<br>㉣ 유기동물 보호센터 '
 "등에서 사육ㆍ관리하는 개(犬) 또는 고양이(猫)</p><br><p id='9' data-category='list' "
 "style='font-size:14px'>사"),
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
 'indexing': {'chunk_id': 'chunk_000006',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
