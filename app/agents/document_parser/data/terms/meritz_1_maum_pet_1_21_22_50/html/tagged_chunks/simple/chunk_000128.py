from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약을 체결할 때 계약에서 정한 피보험자 및 반려동물의 나이에 미달되었거나 초과되<br>었을 경우. 다만, 회사가 나이의 착오를 '
 "발견하였을 때 이미 계약나이에 도달한 경우에<br>는 유효한 계약으로 봅니다.</p><h1 id='30' "
 "style='font-size:14px'>제23조(계약내용의 변경 등)</h1><br><p id='31' "
 "data-category='paragraph' style='font-size:14px'>① 계약자는 회사의 승낙을 얻어 다음의 사항을 "
 '변경할 수 있습니다'),
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
 'indexing': {'chunk_id': 'chunk_000128',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
