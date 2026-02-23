from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만 아래에 기재된 개(犬) 또는 고양이(猫)는 이 보험의 가입 대상이 아<br>닙니다.</p><br><p id='8' "
 "data-category='list' style='font-size:14px'>㉠ 판매점, 브리더 등이 매매(賣買)를 목적으로 "
 '사육ㆍ관리하는 개(犬) 또는 고양이<br>(猫)<br>㉡ 경찰견, 구조견, 군견, 사냥개 등 특수한 목적의 개(犬)(단, 맹도견, 청도견 '
 '등<br>장애인 안내견은 제외) 또는 특수한 목적의 고양이(猫)<br>㉢ 투견, 경주견 등 흥행을 목적으로 사육ㆍ관리하는 개(犬) 또는 '
 '흥행을'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000005',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
