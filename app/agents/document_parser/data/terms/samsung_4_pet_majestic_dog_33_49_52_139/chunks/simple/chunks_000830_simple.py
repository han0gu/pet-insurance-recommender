from langchain_core.documents import Document

chunk = Document(
    page_content=('유사한 구조로 되어 있는 삼륜 또는 사륜의 자동차로서 승용자동차에 해당하지 않 는 자동차\n'
 '3. 전동기를 이용한 동력발생장치를 사용하는 삼륜 또는 사륜의 자동차로서 승용자동 차에 해당하지 않는 자동차\n'
 '④ 제2항 및 제3항에서 자동차관리법(하위 법령, 규칙 포함) 및 도로교통법(하위 법령, 규 칙 포함) 변경시 변경된 내용을 적용합니다. '
 '⑤ 피보험자에게 보험사고가 발생했을 경우 그 사고가 이륜자동차를 운전하는 도중에 발 생한 사고인가 아닌가는 관할 경찰서에서 발행한 '
 '교통사고사실 확인원 등을 주된 판 단자료로 하여 결정합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 132},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000830',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
