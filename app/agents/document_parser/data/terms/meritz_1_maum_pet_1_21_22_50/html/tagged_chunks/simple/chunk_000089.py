from langchain_core.documents import Document

chunk = Document(
    page_content=('계약변경 완료" data-coord="top-left:(222,261); bottom-right:(943,638)" '
 "/></figure><br><p id='111' data-category='list' style='font-size:14px'>③ 회사는 "
 '제2항에 따라 계약내용을 변경할 때 위험이 감소된 경우에는 납입보험료를 감액<br>하고, 이후 기간 보장을 위한 재원인 해약환급금 등의 '
 '차이로 인하여 발생한 정산금액(이<br>하 “정산금액”이라 합니다)을 환급하여 드립니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000089',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
