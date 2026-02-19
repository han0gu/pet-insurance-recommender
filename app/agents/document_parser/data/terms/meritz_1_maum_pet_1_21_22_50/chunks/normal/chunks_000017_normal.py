from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제1항에도 불구하고 보험개시일로부터 그 날을 포함하여 보험증권에 기재된 면책(보상 하지 않는)기간(이하“ 대기기간”) 이내에 발생한 '
 '질병은 보상하지 않습니다. 단, 이 계 약이 갱신계약인 경우에는 적용하지 않습니다. ④ 회사는 제3항 본문의 면책기간을 최대 30일을 '
 '한도로 적용합니다 ⑤ 제1항의 「연간」이라 함은 계약일부터 매 1년 단위로 도래하는 계약해당일 전일까지 의 기간을 말합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 3},
 'term_type': 'basic',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000017',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
