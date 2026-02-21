from langchain_core.documents import Document

chunk = Document(
    page_content=(". 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시</p><br><p id='19' data-category='paragraph' "
 "style='font-size:14px'>② 제1항과 관련하여 통신판매계약의 경우, 회사는 계약자가 가입한 특약만 포함한 "
 '약관을<br>드리며, 전화를 이용하여 체결하는 계약은 계약자의 동의를 얻어 다음의 방법으로 약관<br>의 중요한 내용을 설명할 수 '
 "있습니다.</p><br><p id='20' data-category='paragraph' style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_000119',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
