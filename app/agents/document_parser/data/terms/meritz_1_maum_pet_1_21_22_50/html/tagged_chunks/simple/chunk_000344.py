from langchain_core.documents import Document

chunk = Document(
    page_content=("모두 동일한 보통약관 및 특별약관에 적용됩니<br>다.</p><h1 id='13' "
 "style='font-size:14px'>제2조(특별약관의 체결 및 소멸)</h1><br><p id='14' "
 "data-category='list' style='font-size:14px'>① 이 특약은 계약자의 청약과 보험회사(이하 보험회사는 "
 '「회사」라 합니다)의 승낙으로<br>부가되어집니다.<br>② 제1조(적용대상)의 보험계약이 해지 또는 기타 사유에 의하여 효력을 가지지 '
 '않게 되는<br>경우에는 이 특약은 더 이상 효력을 가지지'),
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
 'indexing': {'chunk_id': 'chunk_000344',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
