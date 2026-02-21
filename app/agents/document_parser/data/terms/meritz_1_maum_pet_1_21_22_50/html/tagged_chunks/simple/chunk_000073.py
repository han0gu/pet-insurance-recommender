from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>현재 시점의 정기예금이율은 보험개발원 홈페이지(www.kidi.or.kr)에서 확인할 수 "
 "있<br>습니다.</p><br><h1 id='86' style='font-size:14px'>【연단위 복리】</h1><br><p "
 "id='87' data-category='paragraph' style='font-size:14px'>회사가 지급할 금전에 이자를 줄 "
 '때, 1년마다 마지막 날에 그 이자를 원금에 더한<br>금액을 다음 1년의 원금으로 하는 이자 계산방법을 말합니다.<br>원금 100원,'),
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
 'indexing': {'chunk_id': 'chunk_000073',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
