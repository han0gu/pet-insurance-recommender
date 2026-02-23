from langchain_core.documents import Document

chunk = Document(
    page_content=('해지권을 행사하는 경우 위의 ‘청구일’은 보험사의 해지 의사표시<br>(서면, 전자우편, 휴대전화 문자메시지 또는 이에 준하는 전자적 '
 "의사표시 포함)가 보험<br>계약자 또는 그의 대리인에게 도달한 날로 봅니다.</p><p id='96' "
 "data-category='paragraph' style='font-size:14px'>제7관 분쟁의 조정 등</p><h1 id='97' "
 "style='font-size:14px'>제34조(분쟁의 조정)</h1><br><p id='98' data-category='list'"),
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
 'indexing': {'chunk_id': 'chunk_000179',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
