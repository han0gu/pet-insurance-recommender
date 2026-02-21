from langchain_core.documents import Document

chunk = Document(
    page_content=('목적의<br>증가 감소 또는 교체) 제2항에도 불구하고 새로이 증가되는 보험의 목적의 보험기간을 계<br>약의 남은 보험기간과 다르게 '
 "정하는 경우에 적용합니다.</p><h1 id='3' style='font-size:14px'>제2조(보험기간)</h1><br><p "
 "id='4' data-category='paragraph' style='font-size:14px'>이 추가특별약관에 따라 계약기간 중에 "
 "새로이 증가된 보험의 목적의 보험기간은 계약자가<br>요청하는 기간으로 합니다.</p><h1 id='5'"),
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
 'indexing': {'chunk_id': 'chunk_000340',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
