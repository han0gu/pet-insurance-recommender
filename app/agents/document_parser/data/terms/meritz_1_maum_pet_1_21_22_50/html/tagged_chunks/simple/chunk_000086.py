from langchain_core.documents import Document

chunk = Document(
    page_content=(". 위 이외에 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때</p><footer id='108' "
 "style='font-size:14px'>- 9 -</footer><p id='109' data-category='paragraph' "
 "style='font-size:14px'>② 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경우에는 제23조(계약내용의 변<br>경 "
 "등)에 따라 계약내용을 변경할 수 있습니다.</p><br><figure id='110'><img style='font-size:14px' "
 'alt="[위험변경에 따른'),
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
 'indexing': {'chunk_id': 'chunk_000086',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
