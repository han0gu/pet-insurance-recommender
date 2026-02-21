from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항에서 정한대로 계약자 또는 보험수익자가 변경내용을 알리지 않은 경우에는 계약\n'
 '자 또는 보험수익자가 회사에 알린 최종의 주소 또는 연락처로 등기우편 등 우편물에\n'
 '대한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 도달에 필요한 기간이 지\n'
 '난 때에 계약자 또는 보험수익자에게 도달된 것으로 봅니다.- 8 -제13조(보험수익자의 지정)보험수익자를 지정하지 않은 때에는 '
 '보험수익자를 피보험자로 합니다.제14조(대표자의 지정)① 계약자 또는 보험수익자가 2명 이상인 경우에는 각 대표자를 1명 지정하여야 '
 '합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000045',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
